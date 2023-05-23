#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda list" after running this.

set -e

# create env
conda create -n medarc python=3.10.8 -y
conda activate medarc

conda install numpy matplotlib tqdm scikit-image -y
conda install -c conda-forge accelerate -y

pip install clip-retrieval webdataset clip pandas matplotlib ftfy regex kornia umap-learn
pip install dalle2-pytorch

pip install torchvision==0.15.2 torch==2.0.1
pip install git+https://github.com/huggingface/diffusers