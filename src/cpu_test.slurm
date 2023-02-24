#!/bin/bash
#SBATCH --account=medarc
#SBATCH --partition=cpu16
#SBATCH --mem=8G
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc

# Run script in singularity container
cd /fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/src
srun singularity exec --bind /fsx:/fsx -H /fsx/home-paulscotti:/home --nv /fsx/home-paulscotti/pytorch_22.08-py3.sif python3 train_prior_w_voxel2clip.py --comment medarc
