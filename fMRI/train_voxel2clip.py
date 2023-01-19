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
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
from info_nce import InfoNCE

import ddp_config
import utils
from models import Clipper, BrainNetwork, NewVoxel3dConvEncoder

if __name__ == '__main__':
    model_name = "voxel2clip"
    outdir = os.path.expanduser(f'~/data/neuro/{model_name}')
    modality = "image" # ("image", "text")
    clip_variant = "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
    norm_embs = True # l2 normalize embeddings
    clamp_embs = False # clamp embeddings to (-1.5, 1.5)
    img_augmenting = True # augment images with random crops
    soft_clip = False

    seed = 0
    batch_size = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    num_devices = torch.cuda.device_count()
    num_workers = num_devices
    num_epochs = 60
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-4
    wandb_log = True
    wandb_project = 'laion-fmri'
    wandb_run_name = f'{model_name}-{str(time.time())}'
    wandb_notes = ""
    first_batch = False
    ckpt_saving = True
    ckpt_interval = None

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    
    print('config:')
    print(json.dumps(config, indent=2))

    utils.seed_everything(seed)

    assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32")
    
    if modality == "text":
        image_var = "trail"
    elif modality == "image":
        image_var = "images"
    else:
        raise Exception(f"Unknown modality: {modality}")

    # setup multi-gpu Data Distributed Processing (ddp) if available
    # if not using ddp, using_ddp should be False and local_rank=0
    using_ddp, local_rank = ddp_config.ddp_test()
    if device == 'cuda':
        torch.cuda.set_device(local_rank)

    # TODO: only on master process
    os.makedirs(outdir, exist_ok=True)

    # load clipper
    clip_extractor = Clipper(clip_variant, clamp_embs=clamp_embs, norm_embs=norm_embs)

    # # load COCO annotations curated in the same way as the mind_reader (Lin Sprague Singh) preprint
    # f = h5py.File('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_subj_indices.hdf5', 'r')
    # subj01_order = f['subj01'][:]
    # f.close()
    # annots = np.load('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy',allow_pickle=True)
    # subj01_annots = annots[subj01_order]

    train_dl, val_dl = utils.get_dataloaders(batch_size, image_var, num_workers=num_workers)

    # get first batches
    for train_i, (voxel0, image0) in enumerate(train_dl):
        break
    for val_i, (val_voxel0, val_image0) in enumerate(val_dl):
        break

    # voxel2clip mapper model
    brain_net = BrainNetwork(768) 
    if using_ddp:
        brain_net0 = brain_net.to(local_rank)
        brain_net = DDP(brain_net0, device_ids=[local_rank])
    else:
        brain_net = brain_net.to(device)

    # Loading checkpoint
    # print("ckpt_path",  ckpt_path)
    # checkpoint = torch.load(ckpt_path, map_location=device)    
    # if 'model_state_dict' in checkpoint:
    #     brain_net.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     brain_net.load_state_dict(checkpoint)
    # brain_net.eval()
    # brain_net.requires_grad_(False)

    def save_ckpt(tag):
        ckpt_path = os.path.join(outdir, f'ckpt-{tag}.pth')
        print(f'saving {ckpt_path}')
        if (using_ddp==False) or (using_ddp==True and local_rank==0):
            state_dict = brain_net.state_dict()
            if using_ddp: # if using DDP, convert DDP state_dict to non-DDP before saving
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
                'sims': sims,
                'val_sims': val_sims,
                }, ckpt_path)
            
            if using_ddp:
                # this tells the other gpus wait for the first gpu to finish saving the model
                dist.barrier()

    utils.count_params(brain_net)

    optimizer = torch.optim.AdamW(brain_net.parameters(), lr=initial_lr)
    if lr_scheduler == 'fixed':
        lr_scheduler = None
    elif lr_scheduler == 'cycle':
        # <TODO> hard-coded values
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            total_steps=num_epochs*((24983//batch_size)//num_devices), 
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )

    epoch = 0
    losses, val_losses, lrs = [], [], []
    sims, val_sims = [], []
    best_val_loss = 1e9
    nce = InfoNCE(temperature=0.01)

    # resume from checkpoint:
    # prior_checkpoint = torch.load(ckpt_path, map_location=device)
    # brain_net.load_state_dict(prior_checkpoint['model_state_dict'])
    # optimizer.load_state_dict(prior_checkpoint['optimizer_state_dict'])
    # lr = prior_checkpoint['lr']
    # epoch = prior_checkpoint['epoch']+1
    # losses = prior_checkpoint['train_losses']
    # optimizer.param_groups[0]['lr'] = lr

    if wandb_log:
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            name=wandb_run_name,
            config=config,
            notes=wandb_notes,
        )

    if first_batch:
        # fake DataLoaders with just the first batches
        bs = 5
        train_dl = [(voxel0[:bs], image0[:bs])]
        val_dl = [(val_voxel0[:bs], val_image0[:bs])]

    def check_loss(loss):
        if loss.isnan().any():
            raise ValueError('NaN loss')

    # feed text and images into diffusion prior network
    progress_bar = tqdm(range(epoch, num_epochs), desc='train loop')

    for epoch in progress_bar:
        brain_net.train()
        
        loss_on_aug, loss_off_aug, aug_pairs = [], [], []

        for train_i, (voxel, image) in enumerate(train_dl):
            optimizer.zero_grad()
            image = image.to(device)
            
            with torch.cuda.amp.autocast():
                if image_var=='images': # using images
                    if img_augmenting:
                        img_input = utils.img_augment(img_input)
                    emb = clip_extractor.embed_image(img_input)
                else: 
                    # using text captions of the images 
                    # emb = clip_extractor.embed_curated_annotations(subj01_annots[img_input])
                    raise NotImplementedError()

                emb_ = brain_net(voxel)
                
                # l2 norm before doing cosine similarity
                emb_ = nn.functional.normalize(emb_, dim=-1)
                labels = torch.arange(len(emb)).to(device)

                if soft_clip:
                    if epoch<10:
                        loss = nce(emb_.reshape(len(emb),-1),emb.reshape(len(emb),-1))
                    else:
                        loss = utils.soft_clip_loss(emb_.reshape(len(emb),-1), emb.reshape(len(emb),-1))
                    # loss = nce(emb_.reshape(len(emb),-1),emb.reshape(len(emb),-1)) + \
                    #        soft_clip_loss(emb_.reshape(len(emb),-1), emb.reshape(len(emb),-1))
                else:
                    loss = nce(emb_.reshape(len(emb),-1), emb.reshape(len(emb),-1))
                check_loss(loss)
                fwd_percent_correct = utils.topk(utils.batchwise_cosine_similarity(emb, emb_), labels, k=1)
                bwd_percent_correct = utils.topk(utils.batchwise_cosine_similarity(emb_, emb), labels, k=1)

            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step() 

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            sims.append(F.cosine_similarity(emb, emb_).mean().item())
            
        brain_net.eval()
        for val_i, (val_voxel, val_image) in enumerate(val_dl):    
            with torch.no_grad(): 
                val_image = val_image.to(device)

                clip_embed = brain_net(val_voxel.to(device).float())
                #clip_embed = nn.functional.normalize(clip_embed,dim=-1)
                # clip_embed = clip_extractor.embed_curated_annotations(subj01_annots[voxel])

                image_clip = clip_extractor.embed_image(val_image).float()

                val_loss, val_pred = brain_net(text_embed=clip_embed, image_embed=image_clip)
                check_loss(val_loss)

                val_losses.append(val_loss.item())
                val_sims.append(F.cosine_similarity(image_clip, val_pred).mean().item())
                
                
        if ckpt_saving:
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

        logs = OrderedDict(
            train_loss=np.mean(losses[-(train_i+1):]),
            val_loss=np.mean(val_losses[-(val_i+1):]),
            lr=lrs[-1],
            train_sim=np.mean(sims[-(train_i+1):]),
            val_sim=np.mean(val_sims[-(val_i+1):]),
        )
        progress_bar.set_postfix(**logs)

        logs = {
            "metrics/train/loss": np.mean(losses[-(train_i+1):]),
            "metrics/val/loss": np.mean(val_losses[-(val_i+1):]),
            "metrics/train/lr": lrs[-1],
            "metrics/train/cosine_sim": np.mean(sims[-(train_i+1):]),
            "metrics/val/cosine_sim": np.mean(val_sims[-(val_i+1):]),
            "metrics/train/num_steps": len(losses),
            "metrics/train/loss_on_aug": np.mean(loss_on_aug),
            "metrics/train/loss_off_aug": np.mean(loss_off_aug),
        }

        if wandb_log:
            wandb.log(logs)
            
    if wandb_log:
        wandb.finish()