"""
Train one single network that includes the voxel2clip mapper and the diffusion prior.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}', flush=True)

import ddp_config
distributed, local_rank = ddp_config.ddp_test()
is_master = (not distributed) or (distributed and local_rank == 0)
print(f'ddp_test: distributed: {distributed}, local_rank: {local_rank}, is_master: {is_master}', flush=True)

from info_nce import InfoNCE
from tqdm import tqdm
from collections import OrderedDict
from dalle2_pytorch import DiffusionPriorNetwork

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainSD
from model3d import NewVoxel3dConvEncoder, SimpleVoxel3dConvEncoder
from diffusers import UniPCMultistepScheduler

# -----------------------------------------------------------------------------
# params for this model
model_name = "prior-w-voxel2clip"
modality = "image" # ("image", "text")
clip_variant = "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
clamp_embs = False # clamp embeddings to (-1.5, 1.5)
alpha_schedule = "constant" # ("constant", "linear") - alpha is weight the MSE DP loss
voxel2clip_kwargs = dict(
    arch='brainnet',
    out_dim=768,
)
prior_kwargs = dict(
    pretrained=True,
    network_kwargs=dict(),
    prior_kwargs=dict(),
)
voxel2clip_path = '' # ckpt path for voxel2clip model
voxel_dims = 1 # (1, 3)
use_mixco = False # use mixco on the voxels
n_samples_save = 8 # how many SD samples from train and val to save
sample_interval = 1 # after how many epochs to save samples
save_at_end = False # sample images and save at end of training only
remote_data = False # pull data from huggingface if True
data_commit = '9947586218b6b7c8cab804009ddca5045249a38d' # only applies when remote_data=True
cache_dir = "/tmp/wds-cache" # for WebDatasets
n_cache_recs = 0
combine_models = True # combine voxel2clip and prior into one model and train both end to end
combine_losses = True # when combine_models=True, use two terms in the loss, NCE and MSE
clip_aug_mode = 'none' # ('none', 'x', 'y')
clip_aug_prob = 0.03 # prob of applying augmentation to a batch
sd_scheduler = 'pndms' # scheduler for SD image variation pipeline ('pndms', 'unipcm')
# -----------------------------------------------------------------------------
# params for all models
seed = 0
batch_size = 64
val_batch_size = 64
num_epochs = 60
lr_scheduler = 'cycle'
initial_lr = 1e-3 #3e-5
max_lr = 3e-4
wandb_log = False
wandb_project = 'fMRI-reconstruction-NSD'
wandb_run_name = ''
wandb_notes = ''
first_batch = False # use only the first batch of training and validation data
ckpt_saving = True # enables checkpoint saving
ckpt_interval = 0 # after how many epochs to save a checkpoint (0 = never, will only save best one)
outdir = f'../train_logs/{model_name}/test'

# -----------------------------------------------------------------------------
# read in any command line args or config file values and override the above params
config_keys = [k for k,v in globals().items() if not k.startswith('_') \
               and isinstance(v, (int, float, bool, str, dict))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if is_master:
    print('config:')
    print(json.dumps(config, indent=2))

if modality == "text":
    image_var = "trial"
elif modality == "image":
    image_var = "images"
else:
    raise Exception(f"Unknown modality: {modality}")

assert n_samples_save <= batch_size * 4

if n_samples_save > 0:
    assert n_samples_save >= 2, 'FID will fail if n_samples_save < 2'

if combine_losses:
    assert combine_models, 'combine_losses=True requires combine_models=True'

# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed + local_rank, cudnn_deterministic=False)

# write config
outdir = os.path.expanduser(outdir)
if is_master:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

if is_master: print('Creating Clipper...')
# Don't L2 norm the extracted CLIP embeddings since we want the prior 
# to learn un-normed embeddings for usage with the SD image variation pipeline.
clip_extractor = Clipper(clip_variant, clamp_embs=clamp_embs, norm_embs=False, device=device)

if is_master: print('Creating voxel2clip...')
voxel2clip_arch = voxel2clip_kwargs.pop('arch')
if voxel2clip_arch == 'brainnet':
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    # 134M params
elif voxel2clip_arch == '3dconv':
    voxel2clip = NewVoxel3dConvEncoder(**voxel2clip_kwargs)
    # 58M params for original version
    # 5M params for smaller version
    # Projection input features: 5120
    # param counts:
    # 5,584,448 total
    # 5,584,448 trainable
elif voxel2clip_arch == '3dconv-simple':
    voxel2clip = SimpleVoxel3dConvEncoder(**voxel2clip_kwargs)
else:
    raise Exception(f"Unknown voxel2clip_arch: {voxel2clip_arch}")
if is_master: print(voxel2clip)
if is_master: utils.count_params(voxel2clip)

if not combine_models:
    voxel2clip.to(device)

    # load voxel2clip model weights
    ckpt = torch.load(voxel2clip_path, map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    voxel2clip.load_state_dict(ckpt)

    # freeze when not combining models
    voxel2clip.eval()
    voxel2clip.requires_grad_(False)

if is_master: print('Creating diffusion prior...')
if not prior_kwargs['pretrained']:
    # same as DALLE2-pytorch
    prior_network = DiffusionPriorNetwork(
        **prior_kwargs['network_kwargs'],
    )

    # custom version of DiffusionPrior from DALLE2-pytorch
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        voxel2clip=voxel2clip,
        **prior_kwargs['prior_kwargs'],
    )
else:
    # not using prior_kwargs b/c the model is pretrained
    diffusion_prior = BrainDiffusionPrior.from_pretrained(
        # kwargs for DiffusionPriorNetwork
        dict(),
        # kwargs for DiffusionNetwork
        dict(
            condition_on_text_encodings=False,
            timesteps=1000,
            voxel2clip=voxel2clip if combine_models else None,
        ),
        voxel2clip_path=voxel2clip_path if combine_models else None,
    )

if distributed:
    diffusion_prior = diffusion_prior.to(local_rank)
    diffusion_prior = DDP(diffusion_prior, device_ids=[local_rank])
else:
    diffusion_prior = diffusion_prior.to(device)

if is_master: utils.count_params(diffusion_prior)

if n_samples_save > 0:
    if is_master: print('Creating SD image variation pipeline...')

    def get_sd_pipe(path_or_name):
        return BrainSD.from_pretrained(
            path_or_name,
            revision="v2.0",
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16, # fp16 is fine if we're not training this
        ).to(device)

    try:
        # try to get local copy, removes a network call to HF that can fail when lots of processes make it all at once
        sd_pipe = get_sd_pipe(
            os.path.join(
                os.path.expanduser('~'), 
                ".cache/huggingface/diffusers/models--lambdalabs--sd-image-variations-diffusers/snapshots/a2a13984e57db80adcc9e3f85d568dcccb9b29fc/"
            ))
    except:
        if is_master: print('Downloading SD image variation pipeline...')
        sd_pipe = get_sd_pipe("lambdalabs/sd-image-variations-diffusers")

    # freeze everything, we're just using this for inference
    sd_pipe.unet.eval()
    sd_pipe.unet.requires_grad_(False)

    sd_pipe.vae.eval()
    sd_pipe.vae.requires_grad_(False)

    sd_pipe.image_encoder.eval()
    sd_pipe.image_encoder.requires_grad_(False)
    assert sd_pipe.image_encoder.training == False

    if sd_scheduler == 'pndms':
        # this is the default
        pass
    elif sd_scheduler == 'unipcm':
        sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)
    else:
        raise ValueError(f"Unknown sd_scheduler: {sd_scheduler}")
    
    if is_master: print('sd_pipe.scheduler', sd_pipe.scheduler)

    # disable progress bar
    sd_pipe.set_progress_bar_config(disable=True)

# # load COCO annotations curated in the same way as the mind_reader (Lin Sprague Singh) preprint
# f = h5py.File('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_subj_indices.hdf5', 'r')
# subj01_order = f['subj01'][:]
# f.close()
# annots = np.load('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy',allow_pickle=True)
# subj01_annots = annots[subj01_order]

if remote_data:
    # pull data directly from huggingface
    train_url, val_url = utils.get_huggingface_urls(data_commit)
    meta_url = None # use original counts
else:
    # local paths
    # train_url = "/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_split/train/train_subj01_{0..49}.tar"
    # val_url = "/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_split/val/val_subj01_0.tar"
    # meta_url = None # use original counts
    
    # stability cluster paths
    train_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/train/train_subj01_{0..49}.tar"
    val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/val/val_subj01_0.tar"
    meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset/metadata_subj01.json"

# which to use for the voxels
if voxel_dims == 1:
    voxels_key = 'nsdgeneral.npy'
elif voxel_dims == 3:
    voxels_key = 'wholebrain_3d.npy'
else:
    raise Exception(f"voxel_dims must be 1 or 3, not {voxel_dims}")

num_devices = torch.cuda.device_count()
num_workers = num_devices

train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size, image_var,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    cache_dir=cache_dir,
    n_cache_recs=n_cache_recs,
    voxels_key=voxels_key,
    val_batch_size=val_batch_size,
)

optimizer = torch.optim.AdamW(diffusion_prior.parameters(), lr=initial_lr)
if lr_scheduler == 'fixed':
    lr_scheduler = None
elif lr_scheduler == 'cycle':
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=num_epochs*((num_train//batch_size)//num_devices),
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )

def save_ckpt(tag):
    if is_master:
        ckpt_path = os.path.join(outdir, f'ckpt-{tag}.pth')
        print(f'saving {ckpt_path}', flush=True)
        state_dict = diffusion_prior.state_dict()
        if distributed:
            # if using DDP, convert DDP state_dict to non-DDP before saving
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

        # if distributed:
        #     # this tells the other gpus wait for the first gpu to finish saving the model
        #     dist.barrier()

epoch = 0
losses, val_losses, lrs = [], [], []
best_val_loss = 1e9

if use_mixco:
    contrast_loss = utils.mixco_nce
else:
    contrast_loss = InfoNCE()
if is_master: print('contrast_loss', contrast_loss)

# weight for prior's MSE loss term
if alpha_schedule == 'constant':
    alphas = np.ones(num_epochs) * 0.01
elif alpha_schedule == 'linear':
    alphas = np.linspace(0.01, 0.05, num_epochs, endpoint=True)
else:
    raise ValueError(f'unknown alpha_schedule: {alpha_schedule}')

# for Atom's loss
epoch_temps = np.linspace(0.004, 0.0075, num_epochs-int(0.5*num_epochs), endpoint=True)

if wandb_log and is_master:
    import wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        notes=wandb_notes,
    )

# get first batches (used for generating samples with SD)
for train_i, (voxel0, image0, key0) in enumerate(train_dl):
    break
for val_i, (val_voxel0, val_image0, val_key0) in enumerate(val_dl):
    break

if first_batch:
    # fake DataLoaders with just the first batches
    # bs = batch_size
    # train_dl = [(voxel0[:bs], image0[:bs], key0[:bs])]
    # val_dl = [(val_voxel0[:bs], val_image0[:bs], val_key0[:bs])]
    train_dl = [(voxel0, image0, key0)]
    val_dl = [(val_voxel0, val_image0, val_key0)]

# feed text and images into diffusion prior network
progress_bar = tqdm(range(epoch, num_epochs), desc='train loop', disable=(distributed and local_rank != 0))

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
    loss_on_aug = []
    loss_off_aug = []
    image_aug = None

    alpha = alphas[epoch]
    epoch_temp = epoch_temps[epoch - int(0.5*num_epochs)]

    keys = set()

    for train_i, (voxel, image, key) in enumerate(train_dl):

        optimizer.zero_grad()
        
        image = image.to(device).float()
        voxel = voxel.to(device).float()
        keys.update(key)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            clip_image = clip_extractor.embed_image(image).float()
            
            if use_mixco:
                voxel, perm, betas, select = utils.mixco(voxel)
            
            if combine_models:
                # loss here is MSE for the prior, clip_voxels are voxel2clip outputs
                loss, pred, clip_voxels = diffusion_prior(image_embed=clip_image, voxel=voxel)
                utils.check_loss(loss)

                if combine_losses:
                    # combine losses for contrastive learned voxel2clip mapper and the prior
                    if use_mixco:
                        if epoch < int(0.5*num_epochs):
                            loss_nce = contrast_loss(
                                nn.functional.normalize(clip_voxels, dim=-1), 
                                nn.functional.normalize(clip_image, dim=-1),
                                temp=0.006, perm=perm, betas=betas, select=select,
                            )
                        else:
                            loss_nce = utils.soft_clip_loss(
                                nn.functional.normalize(clip_voxels, dim=-1), 
                                nn.functional.normalize(clip_image, dim=-1),
                                temp=epoch_temp,
                            )
                    else:
                        loss_nce = contrast_loss(
                            nn.functional.normalize(clip_voxels, dim=-1), 
                            nn.functional.normalize(clip_image, dim=-1),
                        )
                    utils.check_loss(loss_nce)

                    loss_nce_sum += loss_nce.item()
                    loss_prior_sum += loss.item()

                    # MSE and NCE are weighted equally at the beginning,
                    # with alpha=0.01 we'll have something like .01*300 + .99*3 = 3 + 3
                    loss = alpha * loss + (1-alpha) * loss_nce
                else:
                    loss_prior_sum += loss.item()
            else:
                # don't train end to end, just use the frozen voxel2clip to get clip_voxels
                clip_voxels = voxel2clip(voxel)
                
                ## can't go higher than 32 due to memory
                # aug_bs = 32
                # clip_voxels = clip_voxels[:aug_bs]
                # clip_image = clip_image[:aug_bs]
                # image = image[:aug_bs]
                # key = key[:aug_bs]

                if clip_aug_mode == 'x':
                    # the target y is fixed, and we will change the input x
                    if random.random() < clip_aug_prob:
                        print('Augmenting x', flush=True)
                        # get an image variation
                        with torch.inference_mode():
                            image_aug = sd_pipe(
                                image=image,
                                width=256,
                                height=256,
                                num_inference_steps=30,
                            )
                        # utils.save_augmented_images(image_aug, key,
                        #                             '/fsx/proj-medarc/fmri/augmented-images/type-x/')

                        # get the CLIP embedding for the variation and use it for x
                        clip_aug = clip_extractor.embed_image(image_aug).float()
                        # rescale clip embeddings to have norm similar to brain embeddings
                        clip_aug = F.normalize(clip_aug, dim=-1) * clip_voxels.norm(p=2, dim=-1).reshape(-1, 1)

                        loss, pred, _ = diffusion_prior(text_embed=clip_aug, image_embed=clip_image)
                        loss_on_aug.append(loss.item())
                    else:
                        loss, pred, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_image)
                        loss_off_aug.append(loss.item())

                elif clip_aug_mode == 'y':
                    # the input x is fixed, and we will change the target y
                    if random.random() < clip_aug_prob:
                        print('Augmenting y', flush=True)
                        _, clip_pred, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_image)

                        # get an image variation
                        with torch.inference_mode():
                            image_aug = sd_pipe(
                                # duplicate the embedding to serve classifier free guidance
                                image_embeddings=torch.cat(
                                    [torch.zeros_like(clip_pred), clip_pred]
                                ).unsqueeze(1),
                                width=256,
                                height=256,
                                num_inference_steps=30,
                            )
                            # utils.save_augmented_images(image_aug, key,
                            #                             '/fsx/proj-medarc/fmri/augmented-images/type-y/')

                        # get the CLIP embedding for the variation and use it for y
                        clip_aug = clip_extractor.embed_image(image_aug).float()

                        loss, pred, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_aug)
                        loss_on_aug.append(loss.item())
                    else:
                        loss, pred, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_image)
                        loss_off_aug.append(loss.item())
                else:
                    loss, pred, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_image)
                utils.check_loss(loss)

            # print('train_i', train_i, 'voxel.shape', voxel.shape, 
            #     'epoch', epoch, 'local_rank', local_rank, 'loss', loss, flush=True)

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
    
    print('len(train_keys)', len(keys), 'local_rank', local_rank, flush=True)
    
    if is_master:
        diffusion_prior.eval()
        keys = set()
        for val_i, (voxel, image, key) in enumerate(val_dl):
            with torch.no_grad():
                image = image.to(device).float()
                voxel = voxel.to(device).float()
                keys.update(key)
    
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    clip_image = clip_extractor.embed_image(image).float()
                    if combine_models:
                        loss, pred, clip_voxels = diffusion_prior(image_embed=clip_image, voxel=voxel) \
                            if not distributed else diffusion_prior.module(image_embed=clip_image, voxel=voxel)
                        utils.check_loss(loss)

                        if combine_losses:
                            if use_mixco:
                                if epoch < int(0.5*num_epochs):
                                    loss_nce = contrast_loss(
                                        nn.functional.normalize(clip_voxels, dim=-1), 
                                        nn.functional.normalize(clip_image, dim=-1),
                                        temp=0.006,
                                    )
                                else:
                                    loss_nce = utils.soft_clip_loss(
                                        nn.functional.normalize(clip_voxels, dim=-1), 
                                        nn.functional.normalize(clip_image, dim=-1),
                                        temp=epoch_temp,
                                    )
                            else:
                                loss_nce = contrast_loss(
                                    nn.functional.normalize(clip_voxels, dim=-1), 
                                    nn.functional.normalize(clip_image, dim=-1),
                                )
                            
                            utils.check_loss(loss_nce)

                            val_loss_nce_sum += loss_nce.item()
                            val_loss_prior_sum += loss.item()

                            val_loss = alpha * loss + (1-alpha) * loss_nce
                        else:
                            val_loss = loss
                            val_loss_prior_sum += loss.item()
                    else:
                        clip_voxels = voxel2clip(voxel)
                        val_loss, pred, _ = diffusion_prior(text_embed=clip_voxels, image_embed=clip_image) \
                            if not distributed else diffusion_prior.module(text_embed=clip_voxels, image_embed=clip_image)

                    print('val_i', val_i, 'voxel.shape', voxel.shape, 
                        'epoch', epoch, 'loss', val_loss, flush=True)
    
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

        print('len(val_keys)', len(keys), flush=True)
        # if not first_batch:
        #     # make sure we got all of the validation samples when we're not using just the first batch
        #     assert len(keys) == num_val, (len(keys), num_val)
        # if epoch == 0:
        #     print('val_keys', keys, flush=True)
        
        if ckpt_saving:
            # save best model
            val_loss = np.mean(val_losses[-(val_i+1):])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}', flush=True)

            # Save model checkpoint every `ckpt_interval` epochs or on the last epoch
            if (ckpt_interval > 0 and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                save_ckpt(f'epoch{epoch:03d}')
        else:
            print('Not saving checkpoints', flush=True)

        logs = {
            "train/loss": np.mean(losses[-(train_i+1):]),
            "val/loss": np.mean(val_losses[-(val_i+1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "train/cos_sim": sims / (train_i + 1),
            "val/cos_sim": val_sims / (val_i + 1),
            "train/cos_sim_base": sims_base / (train_i + 1),
            "val/cos_sim_base": val_sims_base / (val_i + 1),
            "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
            "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
            "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
            "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
            "train/loss_nce": loss_nce_sum / (train_i + 1),
            "train/loss_mse": loss_prior_sum / (train_i + 1),
            "val/loss_nce": val_loss_nce_sum / (val_i + 1),
            "val/loss_mse": val_loss_prior_sum / (val_i + 1),
            "train/alpha": alpha,
            "train/loss_on_aug": np.mean(loss_on_aug),
            "train/loss_off_aug": np.mean(loss_off_aug),
        }

        print('logs before sampling', logs, flush=True)

        # sample some images
        if n_samples_save > 0 and (epoch + 1) % sample_interval == 0:
            print('n_samples_save', n_samples_save, flush=True)
            if (not save_at_end) or (save_at_end and epoch == num_epochs - 1):
                # training
                print("Sampling training images...", flush=True)
                
                if combine_models:
                    voxel2clip_ = diffusion_prior.voxel2clip if not distributed else diffusion_prior.module.voxel2clip
                else:
                    voxel2clip_ = voxel2clip

                grids, train_fid = utils.sample_images(
                    clip_extractor, 
                    voxel2clip_, 
                    sd_pipe, 
                    diffusion_prior if not distributed else diffusion_prior.module,
                    voxel0[:n_samples_save], 
                    image0[:n_samples_save], 
                    seed=42,
                )
                for i, grid in enumerate(grids):
                    grid.save(os.path.join(outdir, f'samples-train-{key0[i]}.png'))
                if wandb_log:
                    logs['train/samples'] = [wandb.Image(grid, caption=key0[i]) for i, grid in enumerate(grids)]
                print('Computing train fid', flush=True)
                logs['train/FID'] = train_fid.compute().item()

                # validation
                print("Sampling validation images...", flush=True)
                grids, val_fid = utils.sample_images(
                    clip_extractor, 
                    voxel2clip_,
                    sd_pipe, 
                    diffusion_prior if not distributed else diffusion_prior.module,
                    val_voxel0[:n_samples_save], 
                    val_image0[:n_samples_save],
                    seed=42,
                )
                for i, grid in enumerate(grids):
                    grid.save(os.path.join(outdir, f'samples-val-{val_key0[i]}.png'))
                if wandb_log:
                    logs['val/samples'] = [wandb.Image(grid, caption=val_key0[i]) for i, grid in enumerate(grids)]
                print('Computing val fid', flush=True)
                logs['val/FID'] = val_fid.compute().item()

                # # save augmented image pairs
                # if n_aug_save > 0 and image_aug is not None:
                #     assert image[0].shape == image_aug[0].shape, 'batch dim does not match'
                #     # two rows: original, augmented
                #     grid = utils.torch_to_Image(
                #         make_grid(torch.cat((
                #             nn.functional.interpolate(image[:n_aug_save], (256,256), mode="area", antialias=False),
                #             nn.functional.interpolate(image_aug[:n_aug_save], (256,256), mode="area", antialias=False)
                #         )), nrow=image[:n_aug_save].shape[0], padding=10)
                #     )
                #     grid.save(os.path.join(outdir, f'augmented-pairs.png'))
                #     if wandb_log:
                #         logs['train/samples-aug'] = wandb.Image(grid)
        
        if wandb_log:
            wandb.log(logs)
        
        print('logs for epoch', epoch, logs, flush=True)

    if distributed:
        dist.barrier()

if wandb_log and is_master:
    wandb.finish()
