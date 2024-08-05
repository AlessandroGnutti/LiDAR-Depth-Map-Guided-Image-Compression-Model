# LiDAR-Depth-Map-Guided-Image-Compression-Model

Accepted to IEEE ICIP 2024.

## Abstract

The incorporation of LiDAR technology into some high-end smartphones has unlocked numerous possibilities across various applications, including photography, image restoration, augmented reality, and more. In this paper, we introduce a novel direction that harnesses LiDAR depth maps to enhance the compression of the corresponding RGB camera images. To the best of our knowledge, this represents the initial exploration in this particular research direction. Specifically, we propose a Transformer-based learned image compression system capable of achieving variable-rate compression using a single model while utilizing the LiDAR depth map as supplementary information for both the encoding and decoding processes. Experimental results demonstrate that integrating LiDAR yields an average PSNR gain of $0.83$ dB and an average bitrate reduction of $16$% as compared to its absence.

## Install

> git clone https://github.com/AlessandroGnutti/LiDAR-Depth-Map-Guided-Image-Compression-Model
> 
> cd LiDAR-Depth-Map-Guided-Image-Compression-Model
> 
> pip install -U pip
> 
> pip install torch torchvision
> 
> pip install pillow==9.2.0
> 
> pip install shapely==1.7.1
> 
> pip install -e .
> 
> pip install timm tqdm click

## Dataset

Our experiments have been carried out on [ARKitScenes Dataset](https://github.com/apple/ARKitScenes) (depth_upsampling).

## Example usage

### Training

### Testing

`python examples/variablerate_with_lidar.py -c config/variablerate_with_lidar.yaml`

## Citation

`@misc{gnutti2024lidardepthmapguided,

      title={LiDAR Depth Map Guided Image Compression Model}, 
      
      author={Alessandro Gnutti and Stefano Della Fiore and Mattia Savardi and Yi-Hsin Chen and Riccardo Leonardi and Wen-Hsiao Peng},
      
      year={2024},
      
      eprint={2401.06517},
      
      archivePrefix={arXiv},
      
      primaryClass={eess.IV},
      
      url={https://arxiv.org/abs/2401.06517}, 
      
}`
