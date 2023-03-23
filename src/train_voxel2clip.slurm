#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --nodes=1             # node count
#SBATCH --ntasks-per-node=1   # with DDP, must equal num of gpus
#SBATCH --cpus-per-task=4     # rule-of-thumb is 4 times number of gpus
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --time=08:00:00       # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc

# # for DDP
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

# activate conda environment
export CONDA_HOME=/fsx/$(whoami)/miniconda3
eval "$(conda shell.bash hook)"
conda activate medical-v1

MODALITY='image'
RUNID="v2c-${MODALITY}-sp2-1gpu"
OUTDIR="../train_logs/models/voxel2clip/${RUNID}"
mkdir -p ${OUTDIR}

srun python train_voxel2clip.py \
--wandb_log=True \
--wandb_run_name=${RUNID} \
--outdir=${OUTDIR} \
--test_is_val=True \
--modality=${MODALITY} \
--distributed=False \
&> ${OUTDIR}/log.txt

MODALITY='text'
RUNID="v2c-${MODALITY}-sp2-1gpu"
OUTDIR="../train_logs/models/voxel2clip/${RUNID}"
mkdir -p ${OUTDIR}

srun python train_voxel2clip.py \
--wandb_log=True \
--wandb_run_name=${RUNID} \
--outdir=${OUTDIR} \
--test_is_val=True \
--modality=${MODALITY} \
--distributed=False \
&> ${OUTDIR}/log.txt
