#!/usr/bin/env python
# coding: utf-8

# This notebook takes brain voxels and maps them to CLIP-space via a contrastive learning to CLIP space + diffusion prior approach.

# In[ ]:


# # convert this notebook to .py such that you can then run it via slurm with "sbatch main.slurm"
# from subprocess import call
# command = "jupyter nbconvert train_prior_w_voxel2clip.ipynb --to python"
# call(command,shell=True)


# # Import packages & functions

# In[3]:


import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from info_nce import InfoNCE
from tqdm import tqdm
from collections import OrderedDict
from dalle2_pytorch import DiffusionPriorNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

import ddp_config
distributed,local_rank = ddp_config.ddp_test()
if device=='cuda': torch.cuda.set_device(local_rank)

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainSD
from model3d import NewVoxel3dConvEncoder


# # Configurations

# In[6]:


model_name = "prior-w-voxel2clip"
modality = "image" # ("image", "text")
image_var = 'images' if modality=='image' else trial
clip_variant = "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
clamp_embs = False # clamp embeddings to (-1.5, 1.5)
timesteps = 1000 # for diffusion prior
alpha_schedule = "constant" # ("constant", "linear") - for weighting the loss
voxel2clip_kwargs = dict(out_dim=768)

voxel_dims = 1 # 1 for flattened 3 for 3d
n_samples_save = 4 # how many SD samples from train and val to save

num_samples = 24983 #number of total samples, check webdataset metadata.json
batch_size = 32
num_epochs = 60
lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 3e-4
first_batch = False
ckpt_saving = True
ckpt_interval = 5
save_at_end = False
outdir = f'../train_logs/models/{model_name}/test'
if not os.path.exists(outdir):
    os.makedirs(outdir)
use_mp = False
remote_data = False
data_commit = '9947586218b6b7c8cab804009ddca5045249a38d'

# if running command line, read in args or config file values and override above params
try:
    config_keys = [k for k,v in globals().items() if not k.startswith('_') \
                   and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
except:
    pass

num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices

wandb_log = False
wandb_project = 'stability'
wandb_run_name = 'pstest'
wandb_notes = ''
if wandb_log: 
    import wandb
    config = {
      "model_name": model_name,
      "modality": modality,
      "clip_variant": clip_variant,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "initial_lr": initial_lr,
      "max_lr": max_lr,
      "lr_scheduler": lr_scheduler,
      "alpha_schedule": alpha_schedule,
      "clamp_embs": clamp_embs,
    }

cache_dir = 'cache'
n_cache_recs = 0


# # Prep models and data loaders

# In[4]:


if local_rank == 0: print('Creating Clipper...')
    
# Don't L2 norm the extracted CLIP embeddings since we want the prior 
# to learn un-normed embeddings for usage with the SD image variation pipeline.
clip_extractor = Clipper(clip_variant, clamp_embs=clamp_embs, norm_embs=False, device=device)

if local_rank == 0: print('Creating voxel2clip...')

if voxel_dims == 1: # 1D data
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    # 134M params
elif voxel_dims == 3: # 3D data
    voxel2clip = NewVoxel3dConvEncoder(**voxel2clip_kwargs)
    # 58M params for original version
    # 5M params for smaller version
    # Projection input features: 5120
    # param counts:
    # 5,584,448 total
    # 5,584,448 trainable
    
try:
    utils.count_params(voxel2clip)
except:
    if local_rank == 0: print('Cannot count params for voxel2clip (probably because it has Lazy layers)')

if local_rank == 0: print('Creating diffusion prior...')

# initializing diffusion prior with https://huggingface.co/nousr/conditioned-prior
assert timesteps == 1000
diffusion_prior = BrainDiffusionPrior.from_pretrained(
    # kwargs for DiffusionPriorNetwork
    dict(),
    # kwargs for DiffusionNetwork
    dict(
        condition_on_text_encodings=False,
        timesteps=timesteps,
        # cond_drop_prob=cond_drop_prob,
        # image_embed_scale=image_embed_scale,
        voxel2clip=voxel2clip,
    ),
)

if distributed:
    diffusion_prior = diffusion_prior.to(local_rank)
    diffusion_prior = DDP(diffusion_prior, device_ids=[local_rank])
else:
    diffusion_prior = diffusion_prior.to(device)
try:
    utils.count_params(diffusion_prior)
except:
    if local_rank == 0: print('Cannot count params for diffusion_prior (probably because it has Lazy layers)')

if local_rank == 0: print('Creating SD image variation pipeline...')
sd_cache_dir = '/fsx/home-paulscotti/.cache/huggingface/diffusers/models--lambdalabs--sd-image-variations-diffusers/snapshots/a2a13984e57db80adcc9e3f85d568dcccb9b29fc'
if not os.path.isdir(sd_cache_dir): # download from huggingface if not already downloaded / cached
    sd_pipe = BrainSD.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", 
        revision="v2.0",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16, # fp16 is fine if we're not training this
    ).to(device)
else:
    sd_pipe = BrainSD.from_pretrained(
        sd_cache_dir, 
        revision="v2.0",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16, # fp16 is fine if we're not training this
    ).to(device)

# freeze everything, we're just using this for inference
sd_pipe.unet.eval()
sd_pipe.unet.requires_grad_(False)

sd_pipe.vae.eval()
sd_pipe.vae.requires_grad_(False)

sd_pipe.image_encoder.eval()
sd_pipe.image_encoder.requires_grad_(False)
assert sd_pipe.image_encoder.training == False

if local_rank == 0: print('Pulling NSD webdataset data...')
if remote_data:
    # pull data directly from huggingface
    train_url, val_url = utils.get_huggingface_urls(data_commit)
else:
    # local paths
    train_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/train/train_subj01_{0..49}.tar"
    val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/val/val_subj01_0.tar"

# which to use for the voxels
if voxel_dims == 1:
    voxels_key = 'nsdgeneral.npy'
elif voxel_dims == 3:
    voxels_key = 'wholebrain_3d.npy'
else:
    raise Exception(f"voxel_dims must be 1 or 3, not {voxel_dims}")

if local_rank == 0: print('Prepping train and validation dataloaders...')
train_dl, val_dl = utils.get_dataloaders(
    batch_size, 
    image_var, 
    num_samples=num_samples,
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    cache_dir=cache_dir,
    n_cache_recs=n_cache_recs,
    voxels_key=voxels_key,
)

optimizer = torch.optim.AdamW(diffusion_prior.parameters(), lr=initial_lr)
if lr_scheduler == 'fixed':
    lr_scheduler = None
elif lr_scheduler == 'cycle':
    # <TODO> hard-coded values
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        total_steps=num_epochs*((num_samples//batch_size)//num_devices), 
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):
    ckpt_path = os.path.join(outdir, f'ckpt-{tag}.pth')
    print(f'saving {ckpt_path}')
    if local_rank==0:
        state_dict = diffusion_prior.state_dict()
        if distributed: # if using DDP, convert DDP state_dict to non-DDP before saving
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict[key]
                    del state_dict[key]
        torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion_prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
            }, ckpt_path)

        # if wandb_log:
        #     wandb.save(ckpt_path)
print("\nDone with model preparations!")


# # Main

# In[ ]:


# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(local_rank, cudnn_deterministic=False)

epoch = 0
losses, val_losses, lrs = [], [], []
best_val_loss = 1e9
nce = utils.mixco_nce  # same as infonce if mixup not used

# weight for prior's MSE loss term
if alpha_schedule == 'constant':
    alphas = np.ones(num_epochs) * 0.01
elif alpha_schedule == 'linear':
    alphas = np.linspace(0.01, 0.05, num_epochs, endpoint=True)
else:
    raise ValueError(f'unknown alpha_schedule: {alpha_schedule}')

if wandb_log:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        notes=wandb_notes,
    )

voxel0 = image0 = val_voxel0 = val_image0 = None

progress_bar = tqdm(range(epoch,num_epochs), ncols=250, disable=(local_rank!=0))
for epoch in progress_bar:
    diffusion_prior.train()

    sims = 0.
    sims_base = 0.
    val_sims = 0.
    val_sims_base = 0.
    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    val_fwd_percent_correct = 0.
    val_bwd_percent_correct = 0.
    loss_nce_sum = 0.
    loss_prior_sum = 0.
    val_loss_nce_sum = 0.
    val_loss_prior_sum = 0.

    alpha = alphas[epoch]

    for train_i, (voxel, image) in enumerate(train_dl):
        optimizer.zero_grad()

        image = image.to(device).float()
        voxel = voxel.to(device).float()
        if image0 is None and local_rank == 0:
            image0 = image.clone()
            voxel0 = voxel.clone()
            print(voxel0.shape)

        with torch.cuda.amp.autocast(enabled=use_mp):
            with torch.cuda.amp.autocast(enabled=True):
                clip_image = clip_extractor.embed_image(image, return_norm=False).float()
            clip_image.to(voxel.dtype)
            loss, pred, clip_voxels = diffusion_prior(image_embed=clip_image, voxel=voxel)
            utils.check_loss(loss)

            loss_nce = nce(
                nn.functional.normalize(clip_voxels, dim=-1), 
                nn.functional.normalize(clip_image, dim=-1),
                distributed=distributed
            )
            utils.check_loss(loss_nce)

            loss_nce_sum += loss_nce.item()
            loss_prior_sum += loss.item()

            # MSE and NCE are weighted equally at the beginning,
            # with alpha=0.01 we'll have something like .01*300 + .99*3 = 3 + 3
            loss = alpha * loss + (1-alpha) * loss_nce

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            # similarity after prior diffusion
            sims += F.cosine_similarity(clip_image, pred).mean().item()
            # baseline similarity before prior diffusion
            sims_base += F.cosine_similarity(clip_image, clip_voxels).mean().item()
            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_voxels)).to(device)
            fwd_percent_correct += utils.topk(
                utils.batchwise_cosine_similarity(clip_image, clip_voxels), labels, k=1
            )
            bwd_percent_correct += utils.topk(
                utils.batchwise_cosine_similarity(clip_voxels, clip_image), labels, k=1
            )

        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    if local_rank==0: 
        diffusion_prior.eval()
        for val_i, (voxel, image) in enumerate(val_dl): 
            for val_i, (voxel, image) in enumerate(val_dl): 
                with torch.no_grad():
                    image = image.to(device).float()
                    voxel = voxel.to(device).float()
                    if val_image0 is None:
                        val_image0 = image.clone()
                        val_voxel0 = voxel.clone()
                    
                    with torch.cuda.amp.autocast():
                        clip_image = clip_extractor.embed_image(image).float()
                        loss, pred, clip_voxels = diffusion_prior(image_embed=clip_image, voxel=voxel)

                        loss_nce = nce(
                            nn.functional.normalize(clip_voxels, dim=-1), 
                            nn.functional.normalize(clip_image, dim=-1),
                        )
                        val_loss_nce_sum += loss_nce.item()
                        val_loss_prior_sum += loss.item()

                        val_loss = alpha * loss + (1-alpha) * loss_nce

                        val_losses.append(val_loss.item())
                        val_sims += F.cosine_similarity(clip_image, pred).mean().item()
                        val_sims_base += F.cosine_similarity(clip_image, clip_voxels).mean().item()
                        labels = torch.arange(len(clip_voxels)).to(device)
                        val_fwd_percent_correct += utils.topk(
                            utils.batchwise_cosine_similarity(clip_image, clip_voxels), labels, k=1
                        )
                        val_bwd_percent_correct += utils.topk(
                            utils.batchwise_cosine_similarity(clip_voxels, clip_image), labels, k=1
                        )

                logs = OrderedDict(
                    train_loss=np.mean(losses[-(train_i+1):]),
                    val_loss=np.mean(val_losses[-(val_i+1):]),
                    lr=lrs[-1],
                    train_sim=sims / (train_i + 1),
                    val_sim=val_sims / (val_i + 1),
                )
                progress_bar.set_postfix(**logs)

            if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
                # save best model
                val_loss = np.mean(val_losses[-(val_i+1):])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_ckpt('best')
                else:
                    print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')

                # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
                if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                    save_ckpt(f'epoch{epoch:03d}')

            logs = {
                "train/loss": np.mean(losses[-(train_i+1):]),
                "val/loss": np.mean(val_losses[-(val_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "train/cosine_sim": sims / (train_i + 1),
                "val/cosine_sim": val_sims / (val_i + 1),
                "train/cosine_sim_base": sims_base / (train_i + 1),
                "val/cosine_sim_base": val_sims_base / (val_i + 1),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                "train/loss_nce": loss_nce_sum / (train_i + 1),
                "train/loss_prior": loss_prior_sum / (train_i + 1),
                "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                "val/loss_prior": val_loss_prior_sum / (val_i + 1),
                "train/alpha": alpha,
            }

            # sample some images
            if sd_pipe is not None:
                if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                    if (not save_at_end and n_samples_save > 0) or (save_at_end and epoch == num_epochs - 1):
                        # training
                        grids = utils.sample_images(
                            clip_extractor, diffusion_prior.voxel2clip, sd_pipe, diffusion_prior,
                            voxel0[:n_samples_save], image0[:n_samples_save], seed=42,
                        )
                        for i, grid in enumerate(grids):
                            grid.save(os.path.join(outdir, f'samples-train-{i:03d}.png'))
                        if wandb_log:
                            logs['train/samples'] = [wandb.Image(grid) for grid in grids]

                        # validation
                        grids = utils.sample_images(
                            clip_extractor, diffusion_prior.voxel2clip, sd_pipe, diffusion_prior,
                            val_voxel0[:n_samples_save], val_image0[:n_samples_save], seed=42,
                        )
                        for i, grid in enumerate(grids):
                            grid.save(os.path.join(outdir, f'samples-val-{i:03d}.png'))
                        if wandb_log:
                            logs['val/samples'] = [wandb.Image(grid) for grid in grids]

            if wandb_log:
                wandb.log(logs)
    if distributed:
        dist.barrier()

if wandb_log:
    wandb.finish()


# In[ ]:





# In[ ]:




