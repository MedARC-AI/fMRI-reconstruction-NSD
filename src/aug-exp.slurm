#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40
#SBATCH --nodes=1             # node count
#SBATCH --ntasks-per-node=8   # with DDP, must equal num of gpus
#SBATCH --cpus-per-task=4     # rule-of-thumb is 4 times number of gpus
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=40:00:00       # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc
#SBATCH --exclude=ip-26-0-128-[46,48,85,93-94,101,106,111,123,136,142-143,146,168-169,175,183,189,211,215,223,231,244],ip-26-0-129-[0-1,4,6,11,45,48,60,81-82,84-85,94,105,122],ip-26-0-130-[12-13,19,116,127,132,134,147-148,150,163-164,183,193],ip-26-0-131-[4-5,38,51,77,85,89,107-108,111-112,130,143,150-152,168,182-183,188,239-240,244,247],ip-26-0-132-[7,10,21,37,93,98,107,118,130,139,141-142,149,154,184],ip-26-0-133-[67,76,81,89,111,115,126,131-133,140,145,148,151-152,159-160,226,242],ip-26-0-134-[0,26-27,43,52,61,66,76,83,90-91,105,120,134,141,157,201,219,226-227,248,254],ip-26-0-135-[1,4,22,49,55,64,67,110,118,163,173,184,186,190,192-193,204,208,219,242,255],ip-26-0-136-13,ip-26-0-137-[92,94,97,102,115-116,121,124,139,168,175-176,184,196,212,214,240],ip-26-0-138-[3,13,51,62,66,69,71,79,93,101,159,166,171,178,186,188,208,213],ip-26-0-139-[191,200,214,216,218,226,229,235,237,241,246],ip-26-0-141-[140,146,157,161,166,178,217,228,247],ip-26-0-142-[3,13,21,24,29,33,36,38,41,45,49,67,71,103,106,125,144,146,166,184,186,198,204,217,235,237,246,251,254],ip-26-0-143-[30,39,46,53,61,66,111,121,145,164,171,175,180,206,225,230,235,250]

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
# &> /fsx/jimgoo/logs/log-aug-x-v2.txt

srun python train_combined.py \
config/1D_combo.py \
--batch_size=16 \
--val_batch_size=300 \
--wandb_log=True \
--wandb_notes='aug y' \
--outdir=../train_logs/models/prior-w-voxel2clip/aug-exp \
--remote_data=True \
--cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
--n_samples_save=16 \
--sample_interval=10 \
--voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
--combine_models=False \
--combine_losses=False \
--clip_aug_mode='y' \
--clip_aug_prob=0.05 \
--sd_scheduler='unipcm' \
&> /fsx/jimgoo/logs/log-y.txt

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
# &> /fsx/jimgoo/logs/log-n.txt
