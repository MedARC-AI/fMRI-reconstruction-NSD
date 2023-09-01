 # # Import packages & functions

import os
import shutil
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
_, local_rank, device, num_devices = ddp_config.set_ddp()

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

use_reconst = False
if use_reconst:
    batch_size = 8
else:
    batch_size = 32
num_epochs = 50
lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 5e-4
first_batch = False
ckpt_saving = True
ckpt_interval = 24
save_at_end = False
use_mp = False
remote_data = False
subj_id = "01"
mixup_pct = -1
use_cont = True
use_sobel_loss = False
use_blurred_training = False
cont_model = 'cnx'
seed = 42
resume_from_ckpt = False
ups_mode = '4x'

torch.backends.cuda.matmul.allow_tf32 = True
# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(local_rank+seed, cudnn_deterministic=False)

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
    if cont_model == 'cnx':
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

# auto resume
if os.path.exists(os.path.join(outdir, 'last.pth')):
    ckpt_path = os.path.join(outdir, 'last.pth')
    resume_from_ckpt = True

# num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices

cache_dir = 'cache'
n_cache_recs = 0


# # Prep models and data loaders
if local_rank == 0: print('Creating voxel2sd...')

if voxel_dims == 1: # 1D data
    voxel2sd = Voxel2StableDiffusionModel(use_cont=use_cont, in_dim=1685, ups_mode=ups_mode, h=1024, n_blocks=2)
elif voxel_dims == 3: # 3D data
    raise NotImplementedError()
    
voxel2sd.to(device)
voxel2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2sd)
voxel2sd = DDP(voxel2sd, device_ids=[local_rank])

try:
    utils.count_params(voxel2sd)
except:
    if local_rank == 0: print('Cannot count params for voxel2sd (probably because it has Lazy layers)')


if local_rank == 0: print('Pulling NSD webdataset data...')

## USING BOLD5000 DATASET
from torch.utils.data import Dataset, DataLoader
def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

class FMRI_Dataset(Dataset):
    def __init__(self, images, fmri):
        self.images = images
        self.fmri = fmri
        self.coco = fmri
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.fmri[idx], self.images[idx], self.coco[idx]
    
def create_BOLD5000_dataset(path='/fsx/proj-fmri/shared/datasets_god_hcp_bold5000/BOLD5000', fmri_transform=None,
            image_transform=None, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []
    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        # fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))
      
        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)

        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    print("fmri_train_major",fmri_train_major.shape)
    print("fmri_test_major",fmri_test_major.shape)
    print("img_train_major",img_train_major.shape)
    print("img_test_major",img_test_major.shape)
    
    num_voxels = fmri_train_major.shape[-1]
    print("num_voxels", num_voxels)
    return (FMRI_Dataset(img_train_major, fmri_train_major), 
                FMRI_Dataset(img_test_major, fmri_test_major))

### Replacing the NSD dataloaders ###
train_dataset, test_dataset = create_BOLD5000_dataset(subjects = ['CSI1'])
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True)
val_dl = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
god=True
num_train = 4803
num_val = 113
num_voxels = 1685

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
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
    ckpt_path = os.path.join(outdir, f'{tag}.pth')
    if tag == "last":
        if os.path.exists(ckpt_path):
            shutil.copyfile(ckpt_path, os.path.join(outdir, f'{tag}_old.pth'))
    print(f'saving {ckpt_path}')
    if local_rank==0:
        state_dict = voxel2sd.state_dict()
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
    if tag == "last":
        if os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))

        # if wandb_log:
        #     wandb.save(ckpt_path)

# Optionally resume from checkpoint #
if resume_from_ckpt:
    print("\n---resuming from ckpt_path---\n", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint['epoch']+1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
    voxel2sd.module.load_state_dict(checkpoint['model_state_dict'])
    total_steps_done = epoch*((num_train//batch_size)//num_devices)
    for _ in range(total_steps_done):
        lr_scheduler.step()
    del checkpoint
    torch.cuda.empty_cache()
else:
    epoch = 0

if local_rank==0: print("\nDone with model preparations!")

progress_bar = tqdm(range(epoch, num_epochs), ncols=150, disable=(local_rank!=0))
losses = []
val_losses = []
lrs = []
best_val_loss = 1e10
best_ssim = 0
mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)
for epoch in progress_bar:
    train_sampler.set_epoch(epoch)
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

        image = (torch.permute(image, (0,3,1,2)).float()/255).float()

        image = image.to(device).float()
        image_512 = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)
        voxel = voxel.to(device).float()
        if epoch <= mixup_pct * num_epochs:
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
            
            if epoch <= mixup_pct * num_epochs:
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
                sobel_loss = torch.tensor(0)
            

            loss = mse_loss/0.18215 + 2*reconst_loss + 0.1*cont_loss + 16*sobel_loss
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
                image = (torch.permute(image, (0,3,1,2)).float()/255).float()
                image = image.to(device).float()
                image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)              
                voxel = voxel.to(device).float()
                
                with torch.cuda.amp.autocast(enabled=use_mp):
                    image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                    image_enc_pred = voxel2sd.module(voxel)

                    mse_loss = F.mse_loss(image_enc_pred, image_enc)
                    
                    if use_reconst:
                        reconst = autoenc.decode(image_enc_pred[-16:]/0.18215).sample
                        image = image[-16:]
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

            save_ckpt('last')
            # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                save_ckpt(f'epoch{(epoch+1):03d}')
            try:
                orig = image
                if reconst is None:
                    reconst = autoenc.decode(image_enc_pred[-16:].detach()/0.18215).sample
                    orig = image[-16:]
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

    if True:
        dist.barrier()









