## UWSegDepth: Semantic-Aware Instance-Level Depth Estimation in Underwater Scenes
This repository is the PyTorch implementation of UWSegDepth: Semantic-Aware Instance-Level Depth Estimation in Underwater Scenes.

![🔍 Demo: UWSegDepth in Real World Underwater Scenes](assets/Demo.gif)



## Getting Started
```bash
git clone https://github.com/PANpinchi/UWSegDepth.git

cd UWSegDepth
```

## Installation and Setup
#### To setup the virtual environment and install the required packages, use the following commands:
```bash
conda create -n uwsegdepth python=3.10

conda activate uwsegdepth

source install_environment.sh
```
#### or manually execute the following command:
<details>
<summary>📥 <strong>(Optional)</strong> Setup the virtual environment of UWDepth with SADDER. (click to expand)</summary>

#### Run the commands below to manually setup the virtual environment of UWDepth with SADDER:
```bash
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

git clone https://github.com/PANpinchi/SADDER.git

cd ./SADDER

pip install -r dependencies.txt

pip install opencv-contrib-python

pip install tifffile

cd ..
```
</details>

<details>
<summary>📥 <strong>(Optional)</strong> Setup the virtual environment of BARIS-ERA. (click to expand)</summary>

#### Run the commands below to manually setup the virtual environment of BARIS-ERA:
```bash
git clone https://github.com/PANpinchi/BARIS-ERA.git

cd ./BARIS-ERA

pip install -v -e .

pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

pip install terminaltables
pip install pycocotools
pip install scikit-learn
pip install numpy==1.23.5
pip install gdown
pip install mmcls
pip install yapf==0.40.1
pip install natsort

cd ..
```
</details>

## Download the Required Data

#### Download the pretrained model weights by running the following:
```bash
source download_pretrained_weights.sh
```

#### or manually execute the following command:

<details>
<summary>📥 <strong>(Optional)</strong> Download Pre-trained BARIS-ERA Model. (click to expand)</summary>

#### Run the commands below to download the pre-trained BARIS-ERA model:
```bash
mkdir pretrained

cd pretrained

gdown --id 1-nK4MYPiW5bB8wDHbIXzLimRkLLpek6x

gdown --id 1_MxeMnI11CuvWHGEvud7COMwsPyVeNNv

cd ../..
```
Note: `*.pth` files should be placed in the `/pretrained` folder.
</details>

<details>
<summary>📥 <strong>(Optional)</strong> Download Pre-trained CPD Model. (click to expand)</summary>

#### Run the commands below to download the pre-trained CPD model:
```bash
cd ./SADDER/CPD

gdown --id 1Ezqf3rfBbC4iREjE9TfqDt5_QEvBXZ7F

cd ..
```
Note: `CPD-R.pth` files should be placed in the `/CPD` folder.
</details>

<details>
<summary>📥 <strong>(Optional)</strong> Download Pre-trained UDepth Model. (click to expand)</summary>

#### Run the commands below to download the pre-trained UDepth model:
```bash
mkdir saved_udepth_model

cd saved_udepth_model

gdown --id 1VakMGHTAc2b6baEQvijeU2SapClreIYE

gdown --id 1MaNGn8aKYDXrtmuTsaNlhIyk-IeMRJnO

cd ..
```
Note: `*.pth` files should be placed in the `/saved_udepth_model` folder.
</details>

<details>
<summary>📥 <strong>(Optional)</strong> Download Pre-trained UWDepth Model. (click to expand)</summary>

#### Run the commands below to download the pre-trained UWDepth model:
```bash
cd data/saved_models

gdown --id 1oDcUBglz4NvfO3JsyOnqemDffFHHqr3J

gdown --id 14qFV0lR_yDLILSfqr-8d1ajd--gfu-P6

gdown --id 1seBVgaUzDZKMfWBmS0ZMUDo_NdDV0y9B

cd ../..
```
Note: `*.pth` files should be placed in the `/data/saved_models` folder.
</details>

<details>
<summary>📥 <strong>(Optional)</strong> Download Pre-trained UWDepth with SADDER Model. (click to expand)</summary>

#### Run the commands below to download the pre-trained UWDepth with SADDER model:

```bash
cd saved_models

gdown --id 1eqbV9Jq7WCSWd6btxHVD1r2ykMyWLhpe

cd ../..
```
Note: `*.pth` files should be placed in the `/saved_models` folder.
</details>

## Inference
#### Run the commands below to perform our UWSegDepth on video.
```bash
bash run_uwsegdepth.sh
```



## Citation
#### If you use this code, please cite the following:
```bibtex
@misc{pan2025uwsegdepth,
    title  = {UWSegDepth: Semantic-Aware Instance-Level Depth Estimation in Underwater Scenes},
    author = {Pin-Chi Pan and Soo-Chang Pei},
    url    = {https://github.com/PANpinchi/UWSegDepth},
    year   = {2025}
}
```
