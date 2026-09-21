<!--
Modified by Byoungkwon Yoon and contributors, 2025-2026.
Changes document the LiDAR-integrated triangle splatting project and setup.
-->

<h1 align="center">LiDAR-Integrated Coarse-to-Fine Optimization for Geometrically Consistent Triangle Splatting from Mobile Robots</h1>

<div align="center">
  <a href="https://purduelamm.github.io/lidar-ts-page/">Project page</a> &nbsp;|&nbsp;
  <a href="">Arxiv: Coming soon</a>
</div>
<br>

<p align="center">
Byoungkwon Yoon, Hojun Lee, Yuseop Sim, Hangyeom Lee, Dongjun Lee, Martin Byung-Guk Jun
</p>

<br>


This repo contains the official implementation for the paper "LiDAR-Integrated Coarse-to-Fine Optimization for Geometrically Consistent Triangle Splatting from Mobile Robots".


## Installation

The code is tested with Python 3.12 and CUDA 12.8.


Clone the repository
```bash
git clone --recurse-submodules https://github.com/dbss1126/lidar-ts-release.git
cd lidar-ts-release
```

Setup environment
```bash
micromamba create -n lidar-ts python=3.12
micromamba activate lidar-ts
micromamba install cuda -c nvidia/label/cuda-12.8.0


pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Build the CUDA extensions from the repository root:
```bash
conda install -c conda-forge -y glm tbb-devel

python -m pip install ./submodules/diff-triangle2-rasterization --no-build-isolation
```
## Training
To train our model, you can use the following command:
```bash
python train.py -s <path_to_scenes> -m <output_model_path> --eval
```

If you want to train the model on replica scenes, you should add the following command:  
```bash
python train.py -s <path_to_scenes> -m <output_model_path> --room --eval
```

## Create custom PLY files of optimized scenes

To save your optimized scene after training, just run:

```
python create_ply.py <output_model_path>
```

## Dataset

For the dataset that is used in Replica scene, use the link below


Due to license issue, other dataset instructions will be uploaded soon.