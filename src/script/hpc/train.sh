#!/bin/bash

#SBATCH --job-name=fmri-contrastive
#SBATCH --output=fmri-constrastive.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -t 1-1:00:00 
#SBATCH --mem=64g

cd ../..

DATA_DIR="/scratch/yl6624/Data/natural-scences-dataset"
conda activate mindeye
python Train_MindEye.py --data_path $DATA_DIR \
                        --model_name mindeye \
                        --no-wandb_log \
                        --ckpt_interval 1 \
                        --n_samples_save 1 \
                        --no-hidden \
                        --prior
conda deactivate
cd script/hpc