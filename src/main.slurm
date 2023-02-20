#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=g40423
#SBATCH --nodes=1              # node count
#SBATCH --ntasks-per-node=2   # with DDP, must equal num of gpus
#SBATCH --cpus-per-task=8    # rule-of-thumb is 4 times number of gpus
#SBATCH --gres=gpu:2  
#SBATCH --mem-per-gpu=40G     
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc

# for DDP
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Run script in singularity container
cd /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src
srun singularity exec --bind /fsx:/fsx -H /fsx/home-paulscotti:/home --nv /fsx/home-paulscotti/pytorch_22.08-py3.sif python3 train_prior_w_voxel2clip.py --comment medarc
