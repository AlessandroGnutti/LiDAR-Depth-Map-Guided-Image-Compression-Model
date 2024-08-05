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

import math
import random
import torch
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", vimeo=False, sort=False):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        # self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        if not vimeo:
            self.samples = [f for f in splitdir.rglob('*.png')] + [f for f in splitdir.rglob('*.jpg')]
        else:
            self.samples = [f for f in splitdir.rglob('*0.png')]

        self.transform = transform

        if sort:
            self.samples = sorted(self.samples, key=lambda m: int(str(m).split('/')[-1][:-4]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        #img = Image.open(self.samples[index]).convert("RGB")
        img = Image.open(self.samples[index])
        if self.transform:
            return self.transform(img)

    def __len__(self):
        return len(self.samples)


class ImageFolderLidar(Dataset):

    def __init__(self, path1, path2, transform=None):
        dir1 = Path(path1)
        dir2 = Path(path2)

        if not dir1.is_dir():
            raise RuntimeError(f'Invalid directory "{path1}"')
            
        if not dir2.is_dir():
            raise RuntimeError(f'Invalid directory "{path2}"')
      
        self.samples1 = [f for f in dir1.rglob('*.png')] + [f for f in dir1.rglob('*.jpg')]
        
        self.samples2 = [f for f in dir2.rglob('*.png')] + [f for f in dir2.rglob('*.jpg')]
        
        self.samples1 = sorted(self.samples1, key=lambda m: str(m).split('/')[-1][:-4])
        
        self.samples2 = sorted(self.samples2, key=lambda m: str(m).split('/')[-1][:-4])

        self.transform = transform

    def __getitem__(self, index):
        img1 = Image.open(self.samples1[index])
        img2 = Image.open(self.samples2[index])
        
        # normalization before transform
        
        #img2_array = np.array(img2).astype(np.float32);
        #img2_min = img2_array.min();
        #img2_max = img2_array.max();
        
        #img2_mod = Image.fromarray((img2_array-img2_min)/(img2_max-img2_min))
        
        if self.transform:
            tseed = torch.get_rng_state()
            rseed = random.getstate()
            nseed = np.random.get_state()
            
            ret1 = self.transform(img1)
            
            torch.set_rng_state(tseed)
            random.setstate(rseed)
            np.random.set_state(nseed)
            
            ret2 = self.transform(img2)
            
            img2_array = np.array(ret2).astype(np.float32);
            img2_min = img2_array.min();
            img2_max = img2_array.max();
            
            img2_mod = Image.fromarray(np.clip(img2_array/(2**14),0,1))

            #img2_mod = Image.fromarray((img2_array-img2_min)/(img2_max-img2_min))
            
            #if img2_max - img2_min > 0:
            #    img2_mod = Image.fromarray((img2_array-img2_min)/(img2_max-img2_min))
            #elif img2_max > 0:
            #    img2_mod = Image.fromarray(img2_array/img2_max)
            #else:
            #    img2_mod = Image.fromarray(img2_array)
            
            return transforms.ToTensor()(ret1), transforms.ToTensor()(img2_mod)
        else:
            img2_array = np.array(img2).astype(np.float32);
            img2_min = img2_array.min();
            img2_max = img2_array.max();
            
            #if img2_max - img2_min > 0:
            #    img2_mod = Image.fromarray((img2_array-img2_min)/(img2_max-img2_min))
            #elif img2_max > 0:
            #    img2_mod = Image.fromarray(img2_array/img2_max)
            #else:
            #    img2_mod = Image.fromarray(img2_array)
            
            img2_mod = Image.fromarray(np.clip(img2_array/(2**14),0,1))
            
            return transforms.ToTensor()(img1), transforms.ToTensor()(img2_mod)

    def __len__(self):
        return len(self.samples1)


