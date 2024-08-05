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


