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

## x aug
# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=16 \
# --val_batch_size=300 \
# --wandb_log=True \
# --wandb_notes='aug x with prob 0.2' \
# --outdir=../train_logs/models/prior-w-voxel2clip/aug-exp-x-v2 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --n_samples_save=16 \
# --sample_interval=10 \
# --voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
# --combine_models=False \
# --combine_losses=False \
# --clip_aug_mode='x' \
# --clip_aug_prob=0.2 \
# --sd_scheduler='unipcm' \
# &> ../train_logs/logs/log-aug-x.txt

## y aug
# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=16 \
# --val_batch_size=300 \
# --wandb_log=True \
# --wandb_notes='aug y' \
# --outdir=../train_logs/models/prior-w-voxel2clip/aug-exp \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --n_samples_save=16 \
# --sample_interval=10 \
# --voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
# --combine_models=False \
# --combine_losses=False \
# --clip_aug_mode='y' \
# --clip_aug_prob=0.05 \
# --sd_scheduler='unipcm' \
# &> ../train_logs/logs/log-aug-y.txt

## no aug
# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=16 \
# --val_batch_size=300 \
# --wandb_log=True \
# --wandb_notes='aug off' \
# --outdir=../train_logs/models/prior-w-voxel2clip/aug-exp \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --n_samples_save=16 \
# --sample_interval=10 \
# --voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
# --combine_models=False \
# --combine_losses=False \
# --clip_aug_mode='none' \
# --clip_aug_prob=0.0 \
# --sd_scheduler='unipcm' \
# &> ../train_logs/logs/log-aug-n.txt

## no aug, default scheduler, more epochs, log MSE for comparison
srun python train_combined.py \
config/1D_combo.py \
--batch_size=16 \
--val_batch_size=300 \
--wandb_log=True \
--wandb_notes='aug off, default scheduler, log MSE' \
--outdir=../train_logs/models/prior-w-voxel2clip/aug-exp \
--remote_data=True \
--cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
--n_samples_save=16 \
--sample_interval=10 \
--voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
--combine_models=False \
--combine_losses=False \
--clip_aug_mode='none' \
--clip_aug_prob=0.0 \
--num_epochs=100 \
&> ../train_logs/logs/log-aug-n.txt
