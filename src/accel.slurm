#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1      # should = number of gpus
#SBATCH --gres=gpu:1 
#SBATCH --mem=40Gb
#SBATCH --time=02:30:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc
#SBATCH --array=6-256%10

# Set to equal gres=gpu:#! Also add --multi_gpu to srun command if using multi-gpu!
export NUM_GPUS=1

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM=false

export WANDB_DIR="/fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src/wandb/"
export WANDB_CACHE_DIR="/fsx/home-paulscotti/.cache"
export WANDB_MODE="online"

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

source /fsx/home-paulscotti/.bashrc
cd /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src
srun accelerate launch --mixed_precision=fp16 --num_machines $COUNT_NODE --num_processes $(( $NUM_GPUS * $COUNT_NODE )) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --gpu_ids='all' --dynamo_backend='no' /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src/CLIP_to_CLIP_refine.py --model_name="prior_vers_" --wandb_log --voxel2clip_path="../train_logs/v2c_vers/last.pth" --no-use_mixco --num_epochs=75 --n_samples_save=0 --batch_size=86 --no-combine_losses --layer=$SLURM_ARRAY_TASK_ID --versatile



# Brain_to_CLIP.py --model_name="v2c_vers_att" --clip_variant="ViT-L/14" --wandb_log --num_epochs=200 --versatile --batch_size=300

# Brain_to_CLIP.py --model_name="v2c_vers_reshape_256_mse" --clip_variant="ViT-L/14" --wandb_log --num_epochs=200 --versatile --mixup_pct=.33 --batch_size=256 --with_mse --mse_pct=.66

# Brain_to_CLIP_refine.py --model_name="v2c_vers_refined_notnorm_mixco_mult100_3e-4" --clip_variant="ViT-L/14" --wandb_log --num_epochs=200 --versatile --with_mse --mixup_pct=.5 --mse_pct=0. --lr_scheduler="fixed" --initial_lr=3e-4

# Brain_to_CLIP.py --model_name="v2c_vers_mse_" --clip_variant="ViT-L/14" --wandb_log --num_epochs=240 --versatile --with_mse --mixup_pct=.33 --mse_pct=.66


# CLIP_to_CLIP.py --model_name="prior_nomixco_86_nocombinedloss_1gpu" --wandb_log --voxel2clip_path="../train_logs/v2c_avg_v0_partialFalse/best.pth" --no-use_mixco --num_epochs=160 --n_samples_save=4 --batch_size=86 --no-combine_losses


# CLIP_to_CLIP.py --model_name="prior_nomixco_86" --wandb_log --voxel2clip_path="../train_logs/v2c_avg_v0_partialFalse/best.pth" --no-use_mixco --num_epochs=160 --n_samples_save=4 --batch_size=86


# Brain_to_CLIP.py --model_name="v2c_H14_image_mse_mult" --clip_variant="ViT-H-14" --wandb_log --with_mse --mse_mult=4 --num_epochs=180


# --clip_variant="ViT-H-14"


# CLIP_to_CLIP.py --model_name="text_prior_mixco_noaug" --wandb_log --modality="text" --voxel2clip_path="../train_logs/v2c_text/last.pth"

# Brain_to_CLIP.py --model_name="v2c_text" --clip_variant="ViT-L/14" --wandb_log --modality="text"

# CLIP_to_CLIP.py --model_name="prior_nomixco" --wandb_log --voxel2clip_path="../train_logs/v2c_avg_v0_partialFalse/best.pth" --no-use_mixco

# Brain_to_CLIP.py --model_name="v2c_RN50x64" --clip_variant="RN50x64" --wandb_log

# --model_name="v2c_H14_image_nonorm_" --wandb_log=True --num_epochs=120 --mixup_pct=.5

#--multi_gpu 
