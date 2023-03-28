# fMRI-reconstruction-NSD

To create a conda environment that will run the notebooks and training scripts:
```bash
conda env create -f src/environment.yaml
conda activate medical-v1
```
The [setup.sh](./src/setup.sh) script list the conda and pip commands to create this environment. There's also a [Dockerfile](./src/Dockerfile) and docker image that was created with `make build push` on DockerHub at `jimgoo6/laion-fmri`.

To use the pretrained diffusion prior weights from LAION 2B, run the `download.sh` script to get the files from HuggingFace. For more info on how that model was trained, see [https://huggingface.co/nousr/conditioned-prior/](https://huggingface.co/nousr/conditioned-prior/).

## voxel2clip model

The voxel2clip feed forward model takes 1D or 3D voxel vectors and converts to a CLIP vector using contrastive learning. Training is done with the [train_voxel2clip.py](./src/train_voxel2clip.py) script:

```bash
$ python train_voxel2clip.py --help
```
```
usage: train_voxel2clip.py [-h] [--model_name MODEL_NAME] [--modality {image,text}] [--clip_variant {RN50,ViT-L/14,ViT-B/32}] [--outdir OUTDIR] [--wandb_log]
                           [--wandb_project WANDB_PROJECT] [--h5_dir H5_DIR] [--voxel_dims {1,3}] [--remote_data] [--wds_cache_dir WDS_CACHE_DIR] [--disable_image_aug]

Train voxel2clip

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of model, used for wandb logging
  --modality {image,text}
                        image or text
  --clip_variant {RN50,ViT-L/14,ViT-B/32}
                        clip variant
  --outdir OUTDIR       output directory for logs and checkpoints
  --wandb_log           whether to log to wandb
  --wandb_project WANDB_PROJECT
                        wandb project name
  --h5_dir H5_DIR       directory containing COCO h5 files (only used for modality=text)
  --voxel_dims {1,3}    1 for flattened input, 3 for 3d input
  --remote_data         whether to pull data from huggingface
  --wds_cache_dir WDS_CACHE_DIR
                        directory for caching webdatasets fetched from huggingface
  --disable_image_aug   whether to disable image augmentation (only used for modality=image)
```

## Combined model

```bash
# model diagram to go here
```

The combined model is a combination of two models trained end-to-end:

1) The voxel2clip feed forward model that takes 1D or 3D voxel vectors and converts to a CLIP image vector using contrastive learning.
2) The diffusion prior network from DALLE2-pytorch which takes an input CLIP vector and refines it to a target CLIP space using MSE loss. We initialize the weights from the [pretrained prior](https://huggingface.co/nousr/conditioned-prior) that was trained on LAION aesthetics to go from text CLIP to image CLIP as part of the [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch) project.

The loss for this model is a combination of NCE for the voxel2clip model and MSE for the diffusion prior model. There is an alpha parameter that controls the relative weight of the two losses.

To train this model, you use the `train_combined.py` script, which accepts config files and parameter values as arguments. For example, this is the command to run the current best model:

```bash
python train_combined.py \
config/1D_combo.py \
--remote_data=True
```

The `--remote_data=True` flag will download the NSD WebDatasets directly from HuggingFace and save them into `/tmp/wds-cache` by default. To train on the Stability cluster with slurm, run `sbatch train_combined.slurm`, which will essentially run the above command in a conda env with 8 GPUs.

To keep the two models separate and have the voxel2clip model be frozen throughout training, you can set `--combine_models=False` and pass a voxel2clip checkpoint path:

```bash
python train_combined.py \
config/1D_combo.py \
--remote_data=True \
--combine_models=False \
--voxel2clip_path='checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth' \
```

The same `--voxel2clip_path` flag can be used to load a checkpoint during end-to-end training as well (when `--combine_models=True`, which is the default).

When the two models are combined, the default behavior is to combine losses of the two models so that `loss = alpha * loss_mse + loss_nce`. With the default of `alpha = 0.01`, the two terms are weighted roughly equally at the beginning of training. You can disable this via the `--combine_losses=False` flag, which will just use the MSE loss for the diffusion prior model.

## Weights & Biases

Metrics and sampled images are saved to the Stability wandb project at [https://stability.wandb.io/jimgoo/fMRI-reconstruction-NSD?workspace=user-jimgoo](https://stability.wandb.io/jimgoo/fMRI-reconstruction-NSD?workspace=user-jimgoo). 

For logging to the Stability wandb account:

```bash
pip install wandb
wandb login --host=https://stability.wandb.io --relogin
```
