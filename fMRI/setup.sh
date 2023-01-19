#!/bin/bash
#
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for a conda freeze after running this.

set -e

## create env first
# mamba create -n medical-v1 python=3.10 -y

mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

## need at least this version of transformers
mamba install -c huggingface transformers==4.25.1 -y

## using conda will downgrade transformers, so use pip instead
# mamba install -c conda-forge diffusers -y
pip install diffusers[torch]

# ananconda channel must be used so that HF transformers is not downgraded
mamba install -c anaconda ipython ipykernel seaborn pillow h5py scikit-learn -y

pip install webdataset info-nce-pytorch

# for CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

pip install dalle2-pytorch
