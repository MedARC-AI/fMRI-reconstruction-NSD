| Notebook | Colab Link |
| :---: | :---:|
| Guiding VQGAN for fMRI Reconstruction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAION-AI/medical/blob/main/fMRI/explore_and_train_vqgan.ipynb)
| StyleGAN Reconstruction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAION-AI/medical/blob/main/fMRI/stylegan_recon_colab.ipynb)
| Voxel to CLIP-aligned voxels | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAION-AI/medical/blob/main/fMRI/Voxel_to_CLIPvoxel.ipynb)

Currently we are trying to replicate this preprint: https://arxiv.org/abs/2210.01769

## Installation

To create a conda environment that will run the notebooks:
```bash
conda create -n medical-v1 python=3.10
conda activate medical-v1
conda env update -f environment.yaml
# this one must be run individually since it uses git
pip install git+https://github.com/openai/CLIP.git
```
The `setup.sh` script list the conda and pip commands to create this environment.

## Train DiffusionPrior

```bash
python train_prior.py
```

There are some training runs on wandb: https://wandb.ai/jimgoo/laion-fmri

