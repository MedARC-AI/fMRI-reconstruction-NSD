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


import ddp_config
_, local_rank, device = ddp_config.set_ddp()

import utils
from models import Voxel2ViTVQGANModel, Voxel2ViTVQGANTransformerModel

from vitvqgan.vitvqgan import ViTVQ

encoder = {'dim': 768, 'depth': 12,
           'heads': 12, 'mlp_dim': 3072}
decoder = {'dim': 768, 'depth': 12,
           'heads': 12, 'mlp_dim': 3072}
quantizer = {'embed_dim': 32, 'n_embed': 8192}
vitvqgan = ViTVQ(256, 8, encoder, decoder, quantizer, path='../train_logs/models/imagenet_vitvq_base.ckpt').to(device)
# sd = torch.load('../train_logs/models/imagenet_vitvq_base.ckpt', map_location="cpu")["state_dict"]
# vitvqgan.load_state_dict(sd, strict=False)
# except:
#     pass
vitvqgan.eval()
vitvqgan.requires_grad_(False)

train_augs = AugmentationSequential(
    kornia.augmentation.RandomCrop((224, 224), p=0.3),
    kornia.augmentation.Resize((256, 256)),
    kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    kornia.augmentation.RandomGrayscale(p=0.2),
    kornia.augmentation.RandomSolarize(p=0.2),
    kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
    data_keys=["input"],
)

# # Configurations
model_name = "autoencoder"
modality = "image" # ("image", "text")
image_var = 'images' if modality=='image' else None  # trial
clamp_embs = False # clamp embeddings to (-1.5, 1.5)

voxel_dims = 1 # 1 for flattened 3 for 3d
n_samples_save = 4 # how many SD samples from train and val to save

use_reconst = False
if use_reconst:
    batch_size = 4
else:
    batch_size = 32
num_epochs = 120
lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 3e-4
first_batch = False
ckpt_saving = True
ckpt_interval = 5
save_at_end = False
use_mp = False
remote_data = False
data_commit = '9947586218b6b7c8cab804009ddca5045249a38d'
mixup_pct = 0.0
use_cont = True
torch.backends.cuda.matmul.allow_tf32 = True

# if not use_cont:
#     mixup_pct = 0.8

# if running command line, read in args or config file values and override above params
try:
    config_keys = [k for k,v in globals().items() if not k.startswith('_') \
                   and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
except:
    pass

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
if local_rank == 0: print('Creating voxel2vq...')

if voxel_dims == 1: # 1D data
    voxel2vq = Voxel2ViTVQGANTransformerModel() 
    # voxel2vq = Voxel2ViTVQGANModel(use_cont=False)
    # 134M params
elif voxel_dims == 3: # 3D data
    raise NotImplementedError()
    
voxel2vq.to(device)
voxel2vq = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2vq)
# try:
#     voxel2vq.load_state_dict(
#         torch.load('../train_logs/models/clip_image_vitL_2stage_mixco_lotemp_125ep_subj01_best.pth'), 
#         strict=False
#     )
# except:
#     pass
voxel2vq = DDP(voxel2vq, device_ids=[local_rank])

try:
    utils.count_params(voxel2vq)
except:
    if local_rank == 0: print('Cannot count params for voxel2vq (probably because it has Lazy layers)')


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
    image_var,
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    cache_dir=cache_dir,
    n_cache_recs=n_cache_recs,
    voxels_key=voxels_key,
)

no_decay = ['bias']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2vq.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2vq.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
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
        state_dict = voxel2vq.state_dict()
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


# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(local_rank, cudnn_deterministic=False)

if wandb_log:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        notes=wandb_notes,
    )

progress_bar = tqdm(range(num_epochs), ncols=150, disable=(local_rank!=0))
losses = []
val_losses = []
lrs = []
best_val_loss = 1e10
for epoch in progress_bar:
    voxel2vq.train()
    
    loss_mse_sum = 0
    loss_cont_sum = 0
    val_loss_mse_sum = 0
    val_loss_quant_sum = 0
    val_loss_reconst_sum = 0

    for train_i, (voxel, image, _) in enumerate(train_dl):
        optimizer.zero_grad()

        image = image.to(device).float()
        image_aug = train_augs(image)
        voxel = voxel.to(device).float()
        if epoch <= mixup_pct * num_epochs:
            voxel, perm, betas, select = utils.mixco(voxel, beta=0.1)
        else:
            select = None

        with torch.cuda.amp.autocast(enabled=use_mp):
            image_enc = F.normalize(vitvqgan.pre_quant(vitvqgan.encoder(image)), p=2, dim=-1)
            image_enc_q,_,_ = vitvqgan.quantizer(image_enc)
            image_enc_pred = voxel2vq(voxel)
            if epoch <= mixup_pct * num_epochs:
                image_enc_shuf = image_enc[perm]
                betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                    image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)
            
            # doesn't work
            mse_loss = F.mse_loss(image_enc_pred, image_enc)
            cont_loss = F.cross_entropy(
                torch.bmm(image_enc_pred, image_enc_q.permute(0,2,1)), 
                torch.arange(image_enc.shape[1])[None].expand(len(image_enc), -1).to(device)
            )

            loss = mse_loss + cont_loss
            # utils.check_loss(loss)

            loss_mse_sum += mse_loss.item()
            loss_cont_sum += cont_loss.item()

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
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    if local_rank==0: 
        voxel2vq.eval()
        for val_i, (voxel, image, _) in enumerate(val_dl): 
            with torch.no_grad():
                image = image.to(device).float()
                voxel = voxel.to(device).float()
                
                with torch.cuda.amp.autocast(enabled=use_mp):
                    image_enc = F.normalize(vitvqgan.pre_quant(vitvqgan.encoder(image)), p=2, dim=-1)
                    image_enc_aug = F.normalize(vitvqgan.pre_quant(vitvqgan.encoder(image)), p=2, dim=-1)
                    image_enc_pred = voxel2vq.module(voxel)

                    mse_loss = F.mse_loss(image_enc_pred, image_enc)
                    
                    q, q_loss, _ = vitvqgan.quantizer(image_enc_pred)
                    reconst = vitvqgan.decode(q).clamp(0, 1)
                    reconst_loss = F.mse_loss(reconst, image)

                    val_loss_mse_sum += mse_loss.item()
                    val_loss_quant_sum += q_loss.item()
                    val_loss_reconst_sum += reconst_loss.item()

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')

            # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                save_ckpt(f'epoch{(epoch+1):03d}')
                try:
                    orig = image
                    pred_grid = make_grid((reconst*255).byte(), nrow=int(len(reconst)**0.5)).permute(1,2,0).cpu().numpy()
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
            "train/loss_cont": loss_cont_sum / (train_i + 1),
            "val/loss_mse": val_loss_mse_sum / (val_i + 1),
            "val/loss_quant_sum": val_loss_quant_sum / (val_i + 1),
            "val/loss_reconst": val_loss_reconst_sum / (val_i + 1),
        }
        if local_rank==0: print(logs)

        if wandb_log:
            wandb.log(logs)
    if True:
        dist.barrier()

if wandb_log:
    wandb.finish()









