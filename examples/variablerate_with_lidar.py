# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.datasets import ImageFolderLidar
from compressai.zoo import image_models
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import yaml
import numpy as np

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target, mask=  None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        mse = self.mse(output["x_hat"], target)
        out["mse_loss"] =torch.mean(mse)
        out["mse_loss_lambda"] = torch.mean(mask * mse)
        out["rdloss"] = 255**2 * out["mse_loss_lambda"] + out["bpp_loss"]
        out_img = torch.clamp(output["x_hat"],0,1)
        out["psnr"] = self.psnr(out_img, target)
        
        out["ssim"] = ssim(out_img, target, data_range=1, size_average=True)
        out["ms_ssim"] = ms_ssim(out_img, target, data_range=1, size_average=True)
        
        #out["x_hat"] = torch.clamp(output["x_hat"],0,1);
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion_rd, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device
    tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
    #lambda_list = np.array([0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932])
    lambda_list = np.array([0.0001, 0.0004, 0.0009, 0.0018, 0.0035, 0.0067, 0.013, 0.025])
    for i, d in tqdm_emu:
        lidars = d[1].to(device); # all the lidar maps inside i-th batch
        d = d[0].to(device) # all the rgb images inside i-th batch
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        lambda_masks=[]
        masks = []
        lidar_maps = []
        for idx in range(d.shape[0]):
            lmbda_norm = random.random()
            lmbda_norm = (idx%d.shape[0])/d.shape[0]+lmbda_norm*(1/d.shape[0])
            lmbda = np.exp(lmbda_norm*(max(np.log(lambda_list))- min(np.log(lambda_list)))+min(np.log(lambda_list)))
            masks.append(torch.zeros(d.shape[2], d.shape[3]).fill_(lmbda_norm))
            lambda_masks.append(torch.zeros(d.shape[2], d.shape[3]).fill_(lmbda))
            lidar_maps.append(lidars[idx])
            
        lambda_encoder = torch.stack(masks).unsqueeze(1).to(device)
        lidar_map = torch.stack(lidar_maps).to(device)
        
        #roimask = torch.ones_like(mask).to(device)
        lambda_decoder = lambda_encoder[:,:,:d.shape[2]//16,:d.shape[3]//16] 
        lambda_masks = torch.stack(lambda_masks).unsqueeze(1).to(device)
        out_net = model(d, lambda_encoder, lambda_decoder, lidar_map)
        out_criterion = criterion_rd(out_net, d, lambda_masks)
        loss = out_criterion['mse_loss']
        total_loss = out_criterion['rdloss']
        total_loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
        optimizer.step()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        update_txt=f'[{i}/{len(train_dataloader)}] | Loss: {total_loss.item():.3f} | PSNR loss: {out_criterion["psnr"].item():.2f} | MSE loss: {out_criterion["mse_loss"].item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f}'
        #wandb.log({"loss": total_loss.item()})
        #wandb.log({"psnr": out_criterion["psnr"].item()})
        #wandb.log({"mse": out_criterion["mse_loss"].item()})
        #wandb.log({"bpp": out_criterion["bpp_loss"].item()})
        
        tqdm_emu.set_postfix_str(update_txt, refresh=True)

def test_epoch(epoch, test_dataloader, model, criterion_rd, stage='test', tqdm_meter=None):
    model.eval()
    device = next(model.parameters()).device
    #lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932]
    lambda_list = np.array([0.0001, 0.0004, 0.0009, 0.0018, 0.0035, 0.0067, 0.013, 0.025])
    output_images = []
    psnr_list = []
    mse_list = []
    bpp_list = []
    ssim_list = []
    ms_ssim_list = []

    loss_am_mean = AverageMeter()
    with torch.no_grad():
        for n, lmbda in enumerate(lambda_list):
            loss_am = AverageMeter()
            bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            aux_loss = AverageMeter()
            psnr = AverageMeter()
            totalloss = AverageMeter()
            ssim = AverageMeter()
            ms_ssim = AverageMeter()
            
            for i, d in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader),leave=False):
                lidar_map = d[1].to(device); # single lidar map
                d = d[0].to(device) # single rgb image inside
                
                lmbda_norm = (np.log(lmbda)-min(np.log(lambda_list)))/(max(np.log(lambda_list))-min(np.log(lambda_list)))
                lambda_encoder  = torch.zeros(d.shape[0], 1, d.shape[2], d.shape[3], device=device).fill_(lmbda_norm)
                lambda_decoder = lambda_encoder[:,:,:d.shape[2]//16,:d.shape[3]//16]
                lambda_masks  = torch.zeros(d.shape[0], 1, d.shape[2], d.shape[3], device=device).fill_(lmbda)
                
                lidar_map = torch.ones_like(lambda_masks).to(device)
                
                out_net = model(d, lambda_encoder, lambda_decoder, lidar_map)
                out_criterion = criterion_rd(out_net, d, lambda_masks)

                loss = out_criterion['mse_loss'].item()
                total_loss = out_criterion['rdloss'].item()

                aux_loss.update(model.aux_loss().item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                loss_am.update(loss)
                mse_loss.update(out_criterion["mse_loss"].item())
                psnr.update(out_criterion['psnr'].item())
                ssim.update(out_criterion['ssim'].item())
                ms_ssim.update(out_criterion['ms_ssim'].item())
                totalloss.update(total_loss)
                #output_images.append(out_criterion["x_hat"])

            txt = f" {n+1} || Bpp loss: {bpp_loss.avg:.4f} | PSNR: {psnr.avg:.3f} | SSIM: {ssim.avg:.3f} | MS-SSIM: {ms_ssim.avg:.3f} | MSE loss: {mse_loss.avg:.5f} |\n"
            if tqdm_meter:
                tqdm_meter.set_postfix_str(txt)
                
            psnr_list.append(psnr.avg)
            mse_list.append(mse_loss.avg)
            bpp_list.append(bpp_loss.avg)
            ssim_list.append(ssim.avg)
            ms_ssim_list.append(ms_ssim.avg)
            loss_am_mean.update(loss_am.avg)
    
    model.train()
    return loss_am_mean.avg

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    args = parser.parse_args(remaining)
    return args


def main(argv):
    #argv = ["-c", "../config/3_variablerate_with_lidar.yaml"]
    args = parse_args(argv)
    base_dir = init(args)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop((args.patch_size, args.patch_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    train_dataset = ImageFolderLidar(args.dataset_path,args.depth_train, transform=train_transforms)
    test_dataset = ImageFolderLidar(args.kodak_path,args.depth_test, transform=None)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True,pin_memory=(device == "cuda"),)
    test_dataloader = DataLoader(test_dataset,batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    
    
    
    net = image_models[args.model](quality=int(args.quality_level), prompt_config=args, input_resolution=(args.input_resolution, args.input_resolution))
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30],  #[35,70], 
                                                  gamma=0.5)
    rdcriterion = RateDistortionLoss()

    last_epoch = 0
    if args.checkpoint: 
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=True if args.TEST else False)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if args.TEST:
        print(checkpoint['epoch'])
        tqrange = tqdm.trange(last_epoch, args.epochs)
        loss = test_epoch(0, test_dataloader, net, rdcriterion,'test', tqrange)
        return
        
    best_loss = float("inf")
    tqrange = tqdm.trange(last_epoch, args.epochs)
    
    #net = torch.compile(net, backend="aot_eager", mode="reduce-overhead")
    test_epoch(-1, test_dataloader, net, rdcriterion,'val',tqrange)

if __name__ == '__main__':
    main(sys.argv[1:])


