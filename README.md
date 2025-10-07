
![logo](https://github.com/user-attachments/assets/64f089c1-d1d9-4153-ac83-5993424a48d0)

# LiDAR-Depth-Map-Guided-Image-Compression-Model

Accepted to IEEE ICIP 2024.

This study was carried out within the LICAM project – funded by European Union – Next Generation EU  within the PRIN 2022 program (D.D. 104 - 02/02/2022 Ministero dell’Università e della Ricerca).

## Abstract

The incorporation of LiDAR technology into some high-end smartphones has unlocked numerous possibilities across various applications, including photography, image restoration, augmented reality, and more. In this paper, we introduce a novel direction that harnesses LiDAR depth maps to enhance the compression of the corresponding RGB camera images. To the best of our knowledge, this represents the initial exploration in this particular research direction. Specifically, we propose a Transformer-based learned image compression system capable of achieving variable-rate compression using a single model while utilizing the LiDAR depth map as supplementary information for both the encoding and decoding processes. Experimental results demonstrate that integrating LiDAR yields an average PSNR gain of $0.83$ dB and an average bitrate reduction of $16$% as compared to its absence.

## Install

```
git clone https://github.com/AlessandroGnutti/LiDAR-Depth-Map-Guided-Image-Compression-Model
cd LiDAR-Depth-Map-Guided-Image-Compression-Model 
pip install -U pip 
pip install torch torchvision
pip install pillow==9.2.0
pip install shapely==1.7.1 
pip install -e . 
pip install timm tqdm click
```

## Dataset

Our experiments have been carried out on [ARKitScenes Dataset](https://github.com/apple/ARKitScenes) (depth_upsampling). For reference, we have provided an example that includes high-resolution RGB, low-resolution depth, and upsampled low-resolution depth.

## Example usage

### Training

### Testing

`python examples/variablerate_with_lidar.py -c config/variablerate_with_lidar.yaml`

## Pre-trained weights

You can find the pre-trained weights of our proposed method here [download link](https://drive.google.com/file/d/1DcoVQiUGyYBv-NGhTkXi_U7FF8Tsc7tW/view?usp=drive_link).

## Citation

If you use this code in your work, please consider citing the following paper:

```
@INPROCEEDINGS{gnutti2024,
  author={Gnutti, Alessandro and Fiore, Stefano Della and Savardi, Mattia and Chen, Yi-Hsin and Leonardi, Riccardo and Peng, Wen-Hsiao},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Lidar Depth Map Guided Image Compression Model}, 
  year={2024},
  volume={},
  number={},
  pages={1890-1896},
  keywords={Photography;Performance evaluation;Laser radar;Image coding;Cameras;Transformers;Decoding;Learned image compression;LiDAR;depth map;prompts;transformer},
  doi={10.1109/ICIP51287.2024.10648184}}
```

## Contacts

For any information, please send an email to alessandro.gnutti@unibs.it

## Acknowledgement

Our work is based on the framework of [Transformer_VariableROI](https://github.com/NYCU-MAPL/Transformer_VariableROI). We thank the authors for open-sourcing their code.
