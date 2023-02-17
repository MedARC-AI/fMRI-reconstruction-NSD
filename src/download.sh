#!/bin/bash

set -e

mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/jimgoo/fmri-diffusion-prior/resolve/main/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth
wget https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth
wget https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json