#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --nodes=1             # node count
#SBATCH --ntasks-per-node=8   # with DDP, must equal num of gpus
#SBATCH --cpus-per-task=4     # rule-of-thumb is 4 times number of gpus
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=10:00:00       # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc

# for DDP
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# activate conda environment
export CONDA_HOME=/fsx/$(whoami)/miniconda3
eval "$(conda shell.bash hook)"
conda activate medical-v1

# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=64 \
# --val_batch_size=300 \
# --wandb_log=True \
# --wandb_notes='1D e2e (8 GPUs) with pretrained prior AND voxel2clip, MSE loss only' \
# --outdir=../train_logs/models/prior-w-voxel2clip/1D_combo-MSE-only \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --n_samples_save=16 \
# --sample_interval=10 \
# --use_mixco=False \
# --voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
# --combine_losses=False \
# &> ../train_logs/logs/log.txt

## text modality
srun python train_combined.py \
config/1D_combo.py \
--batch_size=64 \
--val_batch_size=300 \
--wandb_log=True \
--wandb_notes='1D e2e (8 GPUs) with pretrained prior, text modality' \
--outdir=../train_logs/models/prior-w-voxel2clip/1D_combo-text \
--remote_data=True \
--cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
--n_samples_save=16 \
--sample_interval=10 \
--modality='text' \
&> ../train_logs/logs/log-text.txt

## same as text above but image modality
# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=64 \
# --val_batch_size=300 \
# --wandb_log=True \
# --wandb_notes='1D e2e (8 GPUs) with pretrained prior, image modality' \
# --outdir=../train_logs/models/prior-w-voxel2clip/1D_combo-image \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --n_samples_save=16 \
# --sample_interval=10 \
# --modality='image' \
# &> ../train_logs/logs/log-image.txt
