#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1      # should = number of gpus
#SBATCH --gres=gpu:1  
#SBATCH --mem=40Gb
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc

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
srun accelerate launch --mixed_precision=fp16 --num_machines $COUNT_NODE --num_processes $(( $NUM_GPUS * $COUNT_NODE )) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --gpu_ids='all' --dynamo_backend='no' /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src/CLIP_to_CLIP.py --model_name="testingg"


# --model_name="v2c_H14_image_nonorm_" --wandb_log=True --num_epochs=120 --mixup_pct=.5

#--multi_gpu 
