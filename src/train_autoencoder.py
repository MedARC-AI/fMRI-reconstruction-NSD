# # Import packages & functions

import os
import sys
import json
import traceback
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

from torchvision.utils import make_grid
from PIL import Image
import kornia
from kornia.augmentation.container import AugmentationSequential
from pytorch_msssim import ssim

import ddp_config
_, local_rank, device = ddp_config.set_ddp()

import utils
from models import Voxel2StableDiffusionModel
from convnext import ConvnextXL


from diffusers.models import AutoencoderKL
autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256
)
autoenc.load_state_dict(torch.load('../train_logs/models/sd_image_var_autoenc.pth'))
autoenc.requires_grad_(False)
autoenc.eval()
autoenc.to(device)

# # Configurations
model_name = "autoencoder"
modality = "image" # ("image", "text")
image_var = 'images' if modality=='image' else None  # trial
clamp_embs = False # clamp embeddings to (-1.5, 1.5)

voxel_dims = 1 # 1 for flattened 3 for 3d
n_samples_save = 4 # how many SD samples from train and val to save

use_reconst = True
if use_reconst:
    batch_size = 8
else:
    batch_size = 32
num_epochs = 120
lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 5e-4
first_batch = False
ckpt_saving = True
ckpt_interval = 5
save_at_end = False
use_mp = False
remote_data = False
data_commit = "avg"  # '9947586218b6b7c8cab804009ddca5045249a38d'
mixup_pct = 0.0
use_cont = True
use_sobel_loss = True
use_blurred_training = True

use_full_trainset = True
subj_id = "01"
seed = 0
# ckpt_path = "../train_logs/models/autoencoder_final/test/ckpt-epoch015.pth"
ckpt_path = None

# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed+local_rank, cudnn_deterministic=False)

torch.backends.cuda.matmul.allow_tf32 = True

# if running command line, read in args or config file values and override above params
try:
    config_keys = [k for k,v in globals().items() if not k.startswith('_') \
                   and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
except:
    pass

if use_cont:
    mixup_pct = -1
    cnx = ConvnextXL('../train_logs/models/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)
    train_augs = AugmentationSequential(
        # kornia.augmentation.RandomCrop((480, 480), p=0.3),
        # kornia.augmentation.Resize((512, 512)),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomSolarize(p=0.2),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
        kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
        data_keys=["input"],
    )

outdir = f'../train_logs/models/{model_name}/test'
if local_rank==0:
    os.makedirs(outdir, exist_ok=True)

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
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "initial_lr": initial_lr,
      "max_lr": max_lr,
      "lr_scheduler": lr_scheduler,
      "clamp_embs": clamp_embs,
    }

cache_dir = 'cache'
n_cache_recs = 0


# # Prep models and data loaders
if local_rank == 0: print('Creating voxel2sd...')

if voxel_dims == 1: # 1D data
    voxel2sd = Voxel2StableDiffusionModel(use_cont=use_cont)
    # 134M params
elif voxel_dims == 3: # 3D data
    raise NotImplementedError()
    
voxel2sd.to(device)
voxel2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2sd)
# try:
#     voxel2sd.load_state_dict(
#         torch.load('/fsx/proj-medarc/fmri/fMRI-reconstruction-NSD/train_logs/models/prior_nodetr_noncausal_posemb_240_cont/epoch239.pth',
#                    map_location=device
#                )['model_state_dict'],
#         strict=False
#     )
# except:
#     pass
voxel2sd = DDP(voxel2sd, device_ids=[local_rank])

try:
    utils.count_params(voxel2sd)
except:
    if local_rank == 0: print('Cannot count params for voxel2sd (probably because it has Lazy layers)')


if local_rank == 0: print('Pulling NSD webdataset data...')
if remote_data:
    # pull data directly from huggingface
    train_url, val_url = utils.get_huggingface_urls(data_commit)
else:
    if data_commit == 'avg':
        train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj{subj_id}_{{0..17}}.tar"
        val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj{subj_id}_0.tar"
        if use_full_trainset:
            train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
            val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
            meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
        elif data_commit == 'indiv':
            train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_indiv_split/train/train_subj{subj_id}_{{0..49}}.tar"
            val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_indiv_split/val/val_subj{subj_id}_0.tar"
        else:
            train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_train/train_subj01_{{0..49}}.tar"
            val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_val/val_subj01_0.tar"
        meta_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj{subj_id}.json"

# which to use for the voxels
if voxel_dims == 1:
    voxels_key = 'nsdgeneral.npy'
elif voxel_dims == 3:
    voxels_key = 'wholebrain_3d.npy'
else:
    raise Exception(f"voxel_dims must be 1 or 3, not {voxel_dims}")

if local_rank == 0: print('Prepping train and validation dataloaders...')
train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size,
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    val_batch_size=16,
    cache_dir='/tmp/wds-cache',
    seed=seed+local_rank,
    voxels_key=voxels_key,
    local_rank=local_rank,
)
if use_full_trainset:
    # combines train and val so meta is not valid anymore
    num_train = 8559 + 300
    num_val = 982

no_decay = ['bias']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2sd.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2sd.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                            total_steps=num_epochs*((num_train//batch_size)//num_devices), 
                                            final_div_factor=1000,
                                            last_epoch=-1, pct_start=2/num_epochs)
    
def save_ckpt(tag):
    ckpt_path = os.path.join(outdir, f'ckpt-{tag}.pth')
    print(f'saving {ckpt_path}')
    if local_rank==0:
        state_dict = voxel2sd.state_dict()
        if True: # if using DDP, convert DDP state_dict to non-DDP before saving
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict[key]
                    del state_dict[key]
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                'lrs': lrs,
                }, ckpt_path)
        except:
            print('Failed to save weights')
            print(traceback.format_exc())

        # if wandb_log:
        #     wandb.save(ckpt_path)
if local_rank==0: print("\nDone with model preparations!")

if wandb_log:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        notes=wandb_notes,
    )

losses = []
val_losses = []
lrs = []
best_val_loss = 1e10
best_ssim = 0
mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)
epoch = 0

if ckpt_path is not None:
    print("\n---resuming from ckpt_path---\n",ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint['epoch']+1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
    voxel2sd.module.load_state_dict(checkpoint['model_state_dict'])
    global_batch_size = batch_size * num_devices
    total_steps_done = epoch*(num_train//global_batch_size)
    for _ in range(total_steps_done):
        lr_scheduler.step()
    del checkpoint
    torch.cuda.empty_cache()

progress_bar = tqdm(range(epoch, num_epochs), ncols=150, disable=(local_rank!=0))

for epoch in progress_bar:
    voxel2sd.train()
    
    loss_mse_sum = 0
    loss_reconst_sum = 0
    loss_cont_sum = 0
    loss_sobel_sum = 0
    val_loss_mse_sum = 0
    val_loss_reconst_sum = 0
    val_ssim_score_sum = 0

    reconst_fails = []

    for train_i, (voxel, image, _) in enumerate(train_dl):
        optimizer.zero_grad()

        image = image.to(device).float()
        image_512 = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)
        voxel = voxel.to(device).float()
        if voxel_dims == 1 and data_commit == 'avg':
                voxel = utils.voxel_select(voxel)
        if epoch < mixup_pct * num_epochs:
            voxel, perm, betas, select = utils.mixco(voxel)
        else:
            select = None

        with torch.cuda.amp.autocast(enabled=use_mp):
            autoenc_image = kornia.filters.median_blur(image_512, (15, 15)) if use_blurred_training else image_512
            image_enc = autoenc.encode(2*autoenc_image-1).latent_dist.mode() * 0.18215
            if use_cont:
                image_enc_pred, transformer_feats = voxel2sd(voxel, return_transformer_feats=True)
            else:
                image_enc_pred = voxel2sd(voxel)
            if epoch < mixup_pct * num_epochs:
                image_enc_shuf = image_enc[perm]
                betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                    image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)
            if use_cont:
                image_norm = (image_512 - mean)/std
                image_aug = (train_augs(image_512) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    F.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    F.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    F.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.075,
                    distributed=True
                )
                del image_aug, cnx_embeds, transformer_feats
            else:
                cont_loss = torch.tensor(0)

            # mse_loss = F.mse_loss(image_enc_pred, image_enc)/0.18215
            mse_loss = F.l1_loss(image_enc_pred, image_enc)
            del image_512, voxel

            if use_reconst: #epoch >= 0.1 * num_epochs:
                # decode only non-mixed images
                if select is not None:
                    selected_inds = torch.where(~select)[0]
                    reconst_select = selected_inds[torch.randperm(len(selected_inds))][:4] 
                else:
                    # reconst_select = torch.randperm(len(image_enc_pred))[:4]
                    reconst_select = torch.arange(len(image_enc_pred))
                image_enc_pred = F.interpolate(image_enc_pred[reconst_select], scale_factor=0.5, mode='bilinear', align_corners=False)
                reconst = autoenc.decode(image_enc_pred/0.18215).sample
                # reconst_loss = F.mse_loss(reconst, 2*image[reconst_select]-1)
                reconst_image = kornia.filters.median_blur(image[reconst_select], (7, 7)) if use_blurred_training else image[reconst_select]
                reconst_loss = F.l1_loss(reconst, 2*reconst_image-1)
                if reconst_loss != reconst_loss:
                    reconst_loss = torch.tensor(0)
                    reconst_fails.append(train_i) 
                if use_sobel_loss:
                    sobel_targ = kornia.filters.sobel(kornia.filters.median_blur(image[reconst_select], (3,3)))
                    sobel_pred = kornia.filters.sobel(reconst/2 + 0.5)
                    sobel_loss = F.l1_loss(sobel_pred, sobel_targ)
                else:
                    sobel_loss = torch.tensor(0)
            else:
                reconst_loss = torch.tensor(0)
            

            loss = mse_loss/0.18215 + 2*reconst_loss + cont_loss + 16*sobel_loss
            # utils.check_loss(loss)

            loss_mse_sum += mse_loss.item()
            loss_reconst_sum += reconst_loss.item()
            loss_cont_sum += cont_loss.item()
            loss_sobel_sum += sobel_loss.item()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if local_rank==0:
                logs = OrderedDict(
                    train_loss=np.mean(losses[-(train_i+1):]),
                    val_loss=np.nan,
                    lr=lrs[-1],
                )
                progress_bar.set_postfix(**logs)
        
        loss.backward()
        # if reconst_loss > 0:
        #     torch.nn.utils.clip_grad_norm_(voxel2sd.parameters(), 1.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    if local_rank==0: 
        voxel2sd.eval()
        for val_i, (voxel, image, _) in enumerate(val_dl): 
            with torch.inference_mode():
                image = image.to(device).float()
                image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)              
                voxel = voxel.to(device).float()
                if voxel_dims == 1 and data_commit == 'avg':
                    voxel = voxel.mean(1)
                
                with torch.cuda.amp.autocast(enabled=use_mp):
                    image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                    image_enc_pred = voxel2sd.module(voxel)

                    mse_loss = F.mse_loss(image_enc_pred, image_enc)
                    
                    if use_reconst:
                        reconst = autoenc.decode(image_enc_pred[:16]/0.18215).sample
                        image = image[:16]
                        reconst_loss = F.mse_loss(reconst, 2*image-1)
                        ssim_score = ssim((reconst/2 + 0.5).clamp(0,1), image, data_range=1, size_average=True, nonnegative_ssim=True)
                    else:
                        reconst = None
                        reconst_loss = torch.tensor(0)
                        ssim_score = torch.tensor(0)

                    val_loss_mse_sum += mse_loss.item()
                    val_loss_reconst_sum += reconst_loss.item()
                    val_ssim_score_sum += ssim_score.item()

                    val_losses.append(mse_loss.item() + reconst_loss.item())        

            logs = OrderedDict(
                train_loss=np.mean(losses[-(train_i+1):]),
                val_loss=np.mean(val_losses[-(val_i+1):]),
                lr=lrs[-1],
            )
            progress_bar.set_postfix(**logs)

        if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
            # save best model
            val_loss = np.mean(val_losses[-(val_i+1):])
            val_ssim = val_ssim_score_sum / (val_i + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                save_ckpt('best_ssim')
            else:
                print(f'not best - val_ssim: {val_ssim:.3f}, best_ssim: {best_ssim:.3f}')

            # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                save_ckpt(f'epoch{(epoch+1):03d}')
                try:
                    orig = image
                    if reconst is None:
                        reconst = autoenc.decode(image_enc_pred[:16].detach()/0.18215).sample
                        orig = image[:16]
                    pred_grid = make_grid(((reconst/2 + 0.5).clamp(0,1)*255).byte(), nrow=int(len(reconst)**0.5)).permute(1,2,0).cpu().numpy()
                    orig_grid = make_grid((orig*255).byte(), nrow=int(len(orig)**0.5)).permute(1,2,0).cpu().numpy()
                    comb_grid = np.concatenate([orig_grid, pred_grid], axis=1)
                    del pred_grid, orig_grid
                    Image.fromarray(comb_grid).save(f'{outdir}/reconst_epoch{(epoch+1):03d}.png')
                except:
                    print("Failed to save reconst image")
                    print(traceback.format_exc())

        logs = {
            "train/loss": np.mean(losses[-(train_i+1):]),
            "val/loss": np.mean(val_losses[-(val_i+1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "train/loss_mse": loss_mse_sum / (train_i + 1),
            "train/loss_reconst": loss_reconst_sum / (train_i + 1),
            "train/loss_cont": loss_cont_sum / (train_i + 1),
            "train/loss_sobel": loss_sobel_sum / (train_i + 1),
            "val/loss_mse": val_loss_mse_sum / (val_i + 1),
            "val/loss_reconst": val_loss_reconst_sum / (val_i + 1),
            "val/ssim": val_ssim_score_sum / (val_i + 1),
        }
        if local_rank==0: print(logs)
        if len(reconst_fails) > 0 and local_rank==0:
            print(f'Reconst fails {len(reconst_fails)}/{train_i}: {reconst_fails}')

        if wandb_log:
            wandb.log(logs)
    if True:
        dist.barrier()

if wandb_log:
    wandb.finish()










