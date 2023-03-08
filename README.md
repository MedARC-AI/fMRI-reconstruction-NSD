# fMRI-reconstruction-NSD

To create a conda environment that will run the notebooks and training scripts:
```bash
conda env create -f src/environment.yaml
conda activate medical-v1
```
The [setup.sh](./src/setup.sh) script list the conda and pip commands to create this environment. There's also a [Dockerfile](./src/Dockerfile) and docker image that was created with `make build push` on DockerHub at `jimgoo6/laion-fmri`.

To use the pretrained diffusion prior weights from LAION 2B, run the `download.sh` script to get the files from HuggingFace. For more info on how that model was trained, see [https://huggingface.co/nousr/conditioned-prior/](https://huggingface.co/nousr/conditioned-prior/).

## Weights & Biases setup

```bash
pip install wandb
wandb login --host=https://stability.wandb.io --relogin
```

## Combined model

```bash
# model diagram to go here
```

The combined model is a combination of two models trained end-to-end:

1) The voxel2clip feed forward model that takes 1D voxel vectors and converts to a CLIP image vector using contrastive learning.
2) The diffusion prior network from DALLE2-pytorch which takes an input CLIP vector and refines it to a target CLIP space using MSE loss. We initialize the weights from the pretrained model trained on LAION aesthetics to go from text CLIP to image CLIP.

The loss for this model is a combination of NCE for the voxel2clip model and MSE for the diffusion prior model. There is an alpha parameter that controls the relative weight of the two losses.

To train this model, you use the `train_combined.py` script, which accepts config files and parameter values as arguments. For example, this is the command to run the current best model:

```bash
python train_combined.py \
config/1D_combo.py \
--batch_size=64 \
--val_batch_size=300 \
--wandb_log=True \
--wandb_notes='1D e2e (8 GPUs)' \
--wandb_project='fMRI-reconstruction-NSD' \
--outdir=../train_logs/models/prior-w-voxel2clip/1D_combo \
--remote_data=True \
--cache_dir='/fsx/proj-medarc/fmri/natural-scenes-dataset/9947586218b6b7c8cab804009ddca5045249a38d' \
--n_samples_save=16 \
--sample_interval=10 \
--use_mixco=False
```

To train on the Stability cluster with slurm, run `sbatch fmri-nds.slurm`, which will essentially run the above command in a conda env with 8 GPUs.

Metrics and sampled images are saved to the Stability wandb project at [https://stability.wandb.io/jimgoo/fMRI-reconstruction-NSD?workspace=user-jimgoo](https://stability.wandb.io/jimgoo/fMRI-reconstruction-NSD?workspace=user-jimgoo). The loss in the first plot is `alpha * loss_mse + loss_nce`. With the default of `alpha = 0.01`, the two terms are weighted roughly equally at the beginning of training.