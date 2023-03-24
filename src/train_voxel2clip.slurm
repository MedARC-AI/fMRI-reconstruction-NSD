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

# activate conda environment
export CONDA_HOME=/fsx/$(whoami)/miniconda3
eval "$(conda shell.bash hook)"
conda activate medical-v1

WANDB_PROJECT='fMRI-reconstruction-NSD'
# H5_DIR='/fsx/proj-medarc/fmri/natural-scenes-dataset/'

MODALITY='text'
RUNID="v2c-${MODALITY}-spAvg-1gpu"
OUTDIR="../train_logs/models/voxel2clip/${RUNID}"
mkdir -p ${OUTDIR}

srun accelerate launch --mixed_precision=fp16 --dynamo_backend='no' \
train_voxel2clip.py \
--model_name=${RUNID} \
--modality=${MODALITY} \
--outdir=${OUTDIR} \
--h5_dir=${H5_DIR} \
--wandb_project=${WANDB_PROJECT} \
--wandb_log=True \
&> ${OUTDIR}/log.txt

# MODALITY='image'
# RUNID="v2c-${MODALITY}-spAvg-1gpu"
# OUTDIR="../train_logs/models/voxel2clip/${RUNID}"
# mkdir -p ${OUTDIR}

# # srun python train_voxel2clip.py \
# # --wandb_log=True \
# # --wandb_run_name=${RUNID} \
# # --outdir=${OUTDIR} \
# # --test_is_val=True \
# # --modality=${MODALITY} \
# # --distributed=False \
# # &> ${OUTDIR}/log.txt

# srun accelerate launch --mixed_precision=fp16 --dynamo_backend='no' \
# train_voxel2clip.py \
# --model_name=${RUNID} \
# --modality=${MODALITY} \
# --outdir=${OUTDIR} \
# --h5_dir=${H5_DIR} \
# --wandb_project=${WANDB_PROJECT} \
# --wandb_log=True \
# &> ${OUTDIR}/log.txt
