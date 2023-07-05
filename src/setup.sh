#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.

set -e

conda create -n mindeye python=3.10.8 -y
conda activate mindeye

conda install numpy matplotlib tqdm scikit-image jupyterlab -y
conda install -c conda-forge accelerate -y

pip install clip-retrieval webdataset clip pandas matplotlib ftfy regex kornia umap-learn
pip install dalle2-pytorch

pip install torchvision==0.15.2 torch==2.0.1
pip install diffusers==0.13.0

pip install info-nce-pytorch==0.1.0
pip install pytorch-msssim