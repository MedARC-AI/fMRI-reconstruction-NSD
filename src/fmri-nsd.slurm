#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40423
#SBATCH --nodes=1             # node count
#SBATCH --ntasks-per-node=1   # with DDP, must equal num of gpus
#SBATCH --cpus-per-task=4     # rule-of-thumb is 4 times number of gpus
#SBATCH --gres=gpu:1
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

# srun python train_prior_w_voxel2clip.py \
# &> /fsx/jimgoo/models/log.txt

## run the 3D model
# srun python train_combined.py \
# config/3D_combo.py \
# --wandb_log=True \
# --outdir=../train_logs/models/prior-w-voxel2clip/3D_combo \
# --batch_size=32 \
# --n_samples_save=16 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=False \
# --first_batch=True \
# --save_at_end=True \
# &> /fsx/jimgoo/logs/log.txt

# ## reproduce the good 1D model
# srun python train_combined.py \
# config/1D_combo.py \
# --wandb_log=True \
# --outdir=../train_logs/models/prior-w-voxel2clip/1D_combo \
# --n_samples_save=8 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=False \
# --sample_interval=10 \
# &> /fsx/jimgoo/logs/log.txt

# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=64 \
# --wandb_log=True \
# --outdir=../train_logs/models/prior-w-voxel2clip/1D_combo \
# --n_samples_save=2 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=True \
# --sample_interval=1 \
# &> /fsx/jimgoo/logs/log.txt

# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=16 \
# --wandb_log=True \
# --outdir=../train_logs/models/prior-w-voxel2clip/1D_combo \
# --n_samples_save=8 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=False \
# --sample_interval=10 \
# --use_mixco=True \
# &> /fsx/jimgoo/logs/log-1D.txt

# srun python train_combined.py \
# config/3D_combo.py \
# --batch_size=16 \
# --wandb_log=True \
# --outdir=../train_logs/models/prior-w-voxel2clip/3D_combo \
# --n_samples_save=8 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=False \
# --sample_interval=20 \
# --use_mixco=True \
# &> /fsx/jimgoo/logs/log-3D.txt

# srun python train_combined.py \
# config/1D_combo.py \
# --batch_size=64 \
# --wandb_log=True \
# --outdir=../train_logs/models/prior-w-voxel2clip/1D_combo \
# --n_samples_save=8 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=False \
# --sample_interval=20 \
# --use_mixco=True \
# &> /fsx/jimgoo/logs/log-1D.txt

# srun python train_combined.py \
# config/3D_combo_simple.py \
# --batch_size=16 \
# --wandb_log=True \
# --wandb_notes='3D without transformer' \
# --outdir=../train_logs/models/prior-w-voxel2clip/3D_combo \
# --n_samples_save=8 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=False \
# --sample_interval=20 \
# --use_mixco=True \
# &> /fsx/jimgoo/logs/log-3D.txt

# srun python train_combined.py \
# config/3D_combo_simple.py \
# --batch_size=16 \
# --wandb_log=True \
# --wandb_notes='3D without transformer' \
# --outdir=../train_logs/models/prior-w-voxel2clip/3D_combo \
# --n_samples_save=8 \
# --remote_data=True \
# --cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
# --ckpt_saving=True \
# --first_batch=False \
# --sample_interval=20 \
# --use_mixco=True \
# &> /fsx/jimgoo/logs/log-3D.txt

srun python train_combined.py \
config/1D_combo.py \
--batch_size=32 \
--wandb_log=False \
--wandb_notes='1D w mixco of soft loss' \
--outdir=../train_logs/models/prior-w-voxel2clip/1D_combo \
--n_samples_save=8 \
--remote_data=True \
--cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
--ckpt_saving=True \
--first_batch=True \
--sample_interval=20 \
--use_mixco=True \
&> /fsx/jimgoo/logs/log-1D.txt
