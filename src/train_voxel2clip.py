# # Import packages & functions

import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import kornia
from kornia.augmentation.container import AugmentationSequential
from tqdm import tqdm
import pandas as pd
import wandb
from collections import OrderedDict
from dalle2_pytorch import DiffusionPriorNetwork #, DiffusionPrior

import ddp_config
import utils
from models import Clipper, BrainNetwork
from model3d import NewVoxel3dConvEncoder

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # params for this model
    model_name = "voxel2clip"
    modality = "image" # ("image", "text")
    image_var = 'images' if modality=='image' else None  # trial
    clip_variant = "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
    clamp_embs = False # clamp embeddings to (-1.5, 1.5)
    dim = 768
    remote_data = False
    data_commit = '9947586218b6b7c8cab804009ddca5045249a38d'
    voxel_dims = 3 # 1 for flattened 3 for 3d
    # -----------------------------------------------------------------------------
    # params for all models
    seed = 0
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    num_devices = torch.cuda.device_count()
    num_workers = num_devices
    num_epochs = 120
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-4
    wandb_log = False
    wandb_project = 'laion-fmri'
    wandb_run_name = ''
    wandb_notes = ''
    first_batch = False
    ckpt_saving = True
    ckpt_interval = 10
    use_mp = False
    distributed = True
    save_at_end = False

    cache_dir = 'cache'
    n_cache_recs = 0
    mixup_pct = 0.5

    torch.backends.cuda.matmul.allow_tf32 = True

    # -----------------------------------------------------------------------------
    try:
        config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
        exec(open('configurator.py').read()) # overrides from command line or config file
        config = {k: globals()[k] for k in config_keys} # will be useful for logging
    except:
        pass

    outdir = os.path.expanduser(f'../train_logs/models/{model_name}/test')
    os.makedirs(outdir, exist_ok=True)
    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    num_workers = num_devices * 4
    if distributed:
        _, local_rank, device = ddp_config.set_ddp()
    else:
        local_rank, device = 0, torch.device('cuda:0')
    
    if local_rank == 0: print('Creating Clipper...')
    
    # Don't L2 norm the extracted CLIP embeddings since we want the prior 
    # to learn un-normed embeddings for usage with the SD image variation pipeline.
    train_augs = AugmentationSequential(
        kornia.augmentation.RandomCrop((140, 140), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
        # random_apply = (1,4)
    )
    clip_extractor = Clipper(clip_variant, clamp_embs=False, norm_embs=False, device=device, train_transforms=train_augs)

    if local_rank == 0: print('Creating voxel2clip...')

    if voxel_dims == 1: # 1D data
        voxel2clip = BrainNetwork(out_dim=dim)
        # 134M params
    elif voxel_dims == 3: # 3D data
        voxel2clip = NewVoxel3dConvEncoder(out_dim=dim)

    if local_rank == 0:
        try:
            utils.count_params(voxel2clip)
        except:
            print('Cannot count params for voxel2clip (probably because it has Lazy layers)')
    
    voxel2clip = voxel2clip.to(device)
    if distributed:
        voxel2clip = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2clip)
        voxel2clip = DDP(voxel2clip, device_ids=[local_rank])

    if local_rank == 0: print('Pulling NSD webdataset data...')
    if remote_data:
        # pull data directly from huggingface
        train_url, val_url = utils.get_huggingface_urls(data_commit)
    else:
        # local paths
        if data_commit is None:
            train_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/train/train_subj01_{0..49}.tar"
            val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/val/val_subj01_0.tar"
        else:
            train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_train/train_subj01_{{0..49}}.tar"
            val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_val/val_subj01_0.tar"

    if voxel_dims == 1:
        voxels_key = 'nsdgeneral.npy'
    elif voxel_dims == 3:
        voxels_key = 'wholebrain_3d.npy'
    else:
        raise Exception(f"voxel_dims must be 1 or 3, not {voxel_dims}")

    if local_rank == 0: print('Prepping train and validation dataloaders...')
    train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
        batch_size, 
        image_var,
        num_devices=num_devices,
        num_workers=num_workers,
        train_url=train_url,
        val_url=val_url,
        cache_dir=cache_dir,
        n_cache_recs=n_cache_recs,
        voxels_key=voxels_key,
        val_batch_size=300
    )

    no_decay = ['bias']
    opt_grouped_parameters = [
        {'params': [p for n, p in voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                                total_steps=num_epochs*((num_train//batch_size)//num_devices), 
                                                final_div_factor=1000,
                                                last_epoch=-1, pct_start=2/num_epochs)

    if local_rank==0: print("\nDone with model preparations!")
    utils.seed_everything(local_rank, cudnn_deterministic=False)

    if wandb_log:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
            notes=wandb_notes,
        )

    progress_bar = tqdm(range(num_epochs), ncols=150, disable=(local_rank!=0))
    epoch = 0
    losses, val_losses, lrs = [], [], []
    best_val_loss = 1e9
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    for epoch in progress_bar:
        voxel2clip.train()

        sims_base = 0.
        val_sims_base = 0.
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        val_fwd_percent_correct = 0.
        val_bwd_percent_correct = 0.

        for train_i, (voxel, image, _) in enumerate(train_dl):
            optimizer.zero_grad()

            image = image.to(device).float()
            voxel = voxel.to(device).float()
            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel.float())

            with torch.cuda.amp.autocast(enabled=use_mp):
                with torch.cuda.amp.autocast(enabled=True):
                    clip_image = clip_extractor.embed_image(image).float()
                clip_image.to(voxel.dtype)
                clip_voxels = voxel2clip(voxel)
                
                labels = torch.arange(len(clip_image)).to(device)
                if epoch < int(mixup_pct * num_epochs):
                    loss = utils.mixco_nce(
                        nn.functional.normalize(clip_voxels, dim=-1), 
                        nn.functional.normalize(clip_image, dim=-1),
                        temp=0.006, perm=perm, betas=betas,
                        select=select, distributed=distributed, local_rank=local_rank
                    )
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss = utils.soft_clip_loss(
                        nn.functional.normalize(clip_voxels, dim=-1), 
                        nn.functional.normalize(clip_image, dim=-1),
                        temp=epoch_temp, distributed=distributed
                    )

                utils.check_loss(loss)

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                sims_base += F.cosine_similarity(clip_image, clip_voxels).mean().item()
                fwd_percent_correct += utils.topk(
                    utils.batchwise_cosine_similarity(clip_image, clip_voxels), labels, k=1
                )
                bwd_percent_correct += utils.topk(
                    utils.batchwise_cosine_similarity(clip_voxels, clip_image), labels, k=1
                )

                if local_rank==0:
                    logs = OrderedDict(
                        train_loss=np.mean(losses[-(train_i+1):]),
                        val_loss=np.nan,
                        lr=lrs[-1],
                        train_sim=sims_base / (train_i + 1),
                        val_sim=np.nan,
                    )
                    progress_bar.set_postfix(**logs)

            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        if local_rank==0: 
            voxel2clip.eval()
            for val_i, (voxel, image, _) in enumerate(val_dl):
                with torch.no_grad():
                    image = image.to(device).float()
                    voxel = voxel.to(device).float()
                    
                    with torch.cuda.amp.autocast(enabled=use_mp):
                        with torch.cuda.amp.autocast():
                            clip_image = clip_extractor.embed_image(image).float()
                        clip_image.to(voxel.dtype)
                        if distributed:
                            clip_voxels = voxel2clip.module(voxel)
                        else:
                            clip_voxels = voxel2clip(voxel)

                        labels = torch.arange(len(clip_image)).to(device)
                        if epoch < int(mixup_pct * num_epochs):
                            loss = utils.mixco_nce(
                                nn.functional.normalize(clip_voxels, dim=-1), 
                                nn.functional.normalize(clip_image, dim=-1),
                                temp=0.006, perm=perm, betas=betas,
                                select=select, distributed=False
                            )
                        else:
                            epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                            loss = utils.soft_clip_loss(
                                nn.functional.normalize(clip_voxels, dim=-1), 
                                nn.functional.normalize(clip_image, dim=-1),
                                temp=epoch_temp, distributed=distributed
                            )

                        val_losses.append(loss.item())
                        val_sims_base += F.cosine_similarity(clip_image, clip_voxels).mean().item()
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
                    train_sim=sims_base / (train_i + 1),
                    val_sim=val_sims_base / (val_i + 1),
                )
                progress_bar.set_postfix(**logs)

            if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
                # save best model
                val_loss = np.mean(val_losses[-(val_i+1):])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.save_ckpt('best')
                else:
                    print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')

                # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
                if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                    utils.save_ckpt(f'epoch{(epoch+1):03d}')

            logs = {
                "train/loss": np.mean(losses[-(train_i+1):]),
                "val/loss": np.mean(val_losses[-(val_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "train/cosine_sim_base": sims_base / (train_i + 1),
                "val/cosine_sim_base": val_sims_base / (val_i + 1),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
            }
            if local_rank==0: print(logs)

            if wandb_log:
                wandb.log(logs)
        if True:
            dist.barrier()

    if wandb_log:
        wandb.finish()