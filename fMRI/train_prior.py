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
from dalle2_pytorch import DiffusionPriorNetwork #, DiffusionPrior

import ddp_config
import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainSD

if __name__ == '__main__':
    
    outdir = os.path.expanduser('~/data/neuro/diffusion-prior')
    modality = "image"
    clip_variant = "ViT-L/14"
    clamp_embs = False # clamp embeddings to (-1.5, 1.5)
    ckpt_path = f'checkpoints/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth'
    dim = 768
    depth = 6
    dim_head = 64
    heads = 12 # heads * dim_head = 12 * 64 = 768
    # timesteps = 1000
    timesteps = 100
    cond_drop_prob = 0.2
    image_embed_scale = None
    # image_embed_scale = 1.0
    condition_on_text_encodings = False
    seed = 0
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    num_devices = torch.cuda.device_count()
    num_workers = num_devices
    num_epochs = 60
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-4
    wandb_log = True
    wandb_project = 'laion-fmri'
    wandb_run_name = 'prior-' + str(time.time())
    wandb_notes = ""
    first_batch = False
    ckpt_saving = True
    ckpt_interval = None
    clip_aug_mode = 'x'
    clip_aug_prob = 0.3
    # how many samples from train and val to save
    n_samples_save = 4
    # how many batches of pairs of (orig, augmented) images to save
    n_aug_save = 1

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    
    print('config:')
    print(json.dumps(config, indent=2))

    utils.seed_everything(seed)

    assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32")
    assert clip_aug_mode in ('x', 'y', 'n')
    
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

    # load SD image variation pipeline
    sd_pipe = BrainSD.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", 
        revision="v2.0",
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    
    assert sd_pipe.image_encoder.training == False
    sd_pipe.unet.eval()
    sd_pipe.unet.requires_grad_(False)
    sd_pipe.vae.eval()
    sd_pipe.vae.requires_grad_(False)

    # load clipper
    clip_extractor = Clipper(clip_variant, clamp_embs=clamp_embs, norm_embs=False)

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
    print("ckpt_path", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)    
    if 'model_state_dict' in checkpoint:
        brain_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        brain_net.load_state_dict(checkpoint)
        
    brain_net.eval()
    brain_net.requires_grad_(False)

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

    # setup prior network
    prior_network = DiffusionPriorNetwork(
        dim=dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads
    ).to(device)

    # custom version that can fix seeds
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=dim,
        condition_on_text_encodings=condition_on_text_encodings,
        timesteps=timesteps,
        cond_drop_prob=cond_drop_prob,
        image_embed_scale=image_embed_scale,
    ).to(device)

    utils.count_params(diffusion_prior)

    optimizer = torch.optim.AdamW(diffusion_prior.parameters(), lr=initial_lr)
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

    # resume from checkpoint:
    # prior_checkpoint = torch.load(ckpt_path, map_location=device)
    # diffusion_prior.load_state_dict(prior_checkpoint['model_state_dict'])
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

    # feed text and images into diffusion prior network
    progress_bar = tqdm(range(epoch, num_epochs), desc='train loop')

    for epoch in progress_bar:
        diffusion_prior.train()
        
        loss_on_aug, loss_off_aug, aug_pairs = [], [], []

        for train_i, (voxel, image) in enumerate(train_dl):
            optimizer.zero_grad()
            image = image.to(device)

            if clip_aug_mode == 'x':
                # the target y is fixed
                image_clip = clip_extractor.embed_image(image).float()

                if random.random() < clip_aug_prob:
                    # get an image variation
                    image_aug = sd_pipe(
                        image=image,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        num_images_per_prompt=1,
                        width=256,
                        height=256,
                    )
                    # get the CLIP embedding for the variation and use it for x
                    clip_embed = clip_extractor.embed_image(image_aug).float()

                    # utils.torch_to_Image(image[0]).save('test-orig.png')
                    # utils.torch_to_Image(image_aug[0]).save('test-aug.png')
                    # import ipdb; ipdb.set_trace()
                    # utils.torch_to_Image(
                    #     make_grid(torch.cat((image, image_aug)), nrow=image.shape[0], padding=10)
                    # ).save('test-aug.png')

                    loss, pred = diffusion_prior(text_embed=clip_embed, image_embed=image_clip)
                    loss_on_aug.append(loss.item())
                    if len(aug_pairs) < n_aug_save:
                        aug_pairs.append((image, image_aug))
                else:
                    clip_embed = brain_net(voxel.to(device).float())
                    loss, pred = diffusion_prior(text_embed=clip_embed, image_embed=image_clip)
                    loss_off_aug.append(loss.item())

            elif clip_aug_mode == 'y':
                # the input x is fixed
                clip_embed = brain_net(voxel.to(device).float())

                if random.random() < clip_aug_prob:
                    # get an image variation
                    image_aug = sd_pipe(
                        # duplicate the embedding to serve classifier free guidance
                        image_embeddings=torch.cat([torch.zeros_like(clip_embed), clip_embed]).unsqueeze(1),
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        num_images_per_prompt=1,
                    )
                    # get the CLIP embedding for the variation and use it for y
                    image_clip = clip_extractor.embed_image(image_aug).float()

                    loss, pred = diffusion_prior(text_embed=clip_embed, image_embed=image_clip)
                    loss_on_aug.append(loss.item())
                    if len(aug_pairs) < n_aug_save:
                        aug_pairs.append((image, image_aug))
                else:
                    image_clip = clip_extractor.embed_image(image).float()
                    loss, pred = diffusion_prior(text_embed=clip_embed, image_embed=image_clip)
                    loss_off_aug.append(loss.item())
            else:
                # the input x is fixed
                clip_embed = brain_net(voxel.to(device).float())
                # the target y is fixed
                image_clip = clip_extractor.embed_image(image).float()
                loss, pred = diffusion_prior(text_embed=clip_embed, image_embed=image_clip)

            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step() 

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            sims.append(F.cosine_similarity(image_clip, pred).mean().item())
            
        diffusion_prior.eval()
        for val_i, (val_voxel, val_image) in enumerate(val_dl):    
            with torch.no_grad(): 
                val_image = val_image.to(device)

                clip_embed = brain_net(val_voxel.to(device).float())
                #clip_embed = nn.functional.normalize(clip_embed,dim=-1)
                # clip_embed = clip_extractor.embed_curated_annotations(subj01_annots[voxel])

                image_clip = clip_extractor.embed_image(val_image).float()

                val_loss, val_pred = diffusion_prior(text_embed=clip_embed, image_embed=image_clip)

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

        if n_samples_save > 0:
        #if epoch == num_epochs - 1:
            grids = utils.sample_images(
                clip_extractor, brain_net, sd_pipe, diffusion_prior,
                voxel0[:n_samples_save], image0[:n_samples_save], seed=42,
            )
            logs['media/train/samples'] = [wandb.Image(grid) for grid in grids]

            grids = utils.sample_images(
                clip_extractor, brain_net, sd_pipe, diffusion_prior,
                val_voxel0[:n_samples_save], val_image0[:n_samples_save], seed=42,
            )
            logs['media/val/samples'] = [wandb.Image(grid) for grid in grids]

            if len(aug_pairs) > 0:
                # import ipdb; ipdb.set_trace()
                imgs_orig, imgs_aug = zip(*aug_pairs)
                imgs_orig = imgs_orig[:8]
                imgs_aug = imgs_aug[:8]
                imgs_orig = nn.functional.interpolate(imgs_orig[0], (256,256), mode="area", antialias=False)
                imgs_aug = nn.functional.interpolate(imgs_aug[0], (256,256), mode="area", antialias=False)
                aug_grid = utils.torch_to_Image(
                    make_grid(torch.cat((imgs_orig, imgs_aug)), nrow=imgs_orig.shape[0], padding=10)
                )
                logs['media/train/augmented-pairs'] = wandb.Image(aug_grid)
            
        if wandb_log:
            wandb.log(logs)
            
    if wandb_log:
        wandb.finish()