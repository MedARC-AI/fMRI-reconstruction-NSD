#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --job-name=test
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1    # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc
#SBATCH --requeue
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out

# Set to equal gres=gpu:#
export NUM_GPUS=1

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

export WANDB_DIR="/fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src/wandb/"
export WANDB_CACHE_DIR="/fsx/home-paulscotti/.cache"
export WANDB_MODE="online"

H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo RANK = $RANK
echo WORLD_SIZE=${COUNT_NODE}

source /fsx/home-paulscotti/.bashrc
cd /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src

accelerate launch --num_processes $(($NUM_GPUS * $COUNT_NODE)) --num_machines $COUNT_NODE --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src/Train_MindEye.py --model_name="test" --clip_variant=ViT-L/14 --batch_size=32 --n_samples_save=0 --max_lr=3e-4 --wandb_log
