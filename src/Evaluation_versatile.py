#!/usr/bin/env python
# coding: utf-8

# # Import packages & functions

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import cv2
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils
from models import BrainNetwork, BrainDiffusionPrior, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

seed=42
utils.seed_everything(seed=seed)

# Load CLIP extractor
from models import Clipper
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
imsize = 512


# # CLIP retrieval evaluation

# ## Load pretrained voxel2clip model

# In[9]:


# Load image token model
model_name = "fsdptest6new" #"v2c_vers___" #"v2c_vers_new_mse5_1e-3" # "v2c_vers"
out_dim = 512 #257 * 768

outdir = f'../train_logs/{model_name}'
ckpt_path = os.path.join(outdir, f'last_backup.pth')
print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("EPOCH: ",checkpoint['epoch'])

with torch.no_grad():
    voxel2clip = BrainNetwork(out_dim=out_dim) 
    voxel2clip.eval()
    voxel2clip.load_state_dict(checkpoint['model_state_dict'],strict=False) 
    voxel2clip.requires_grad_(False) # dont need to calculate gradients
    voxel2clip.to(device)
    pass

del checkpoint


# ## For Alex

# In[6]:


# # Load image token model
# model_name = "v2c_b32_image"
# out_dim = 512

# outdir = f'../train_logs/{model_name}'
# ckpt_path = os.path.join(outdir, f'last_backup.pth')
# print("ckpt_path",ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location='cpu')

# print("EPOCH: ",checkpoint['epoch'])

# with torch.no_grad():
#     voxel2clip = BrainNetwork(out_dim=out_dim) 
#     voxel2clip.eval()
#     voxel2clip.load_state_dict(checkpoint['model_state_dict'],strict=False) 
#     voxel2clip.requires_grad_(False) # dont need to calculate gradients
#     voxel2clip.to(device)
#     pass

# del checkpoint

# train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
# val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
# meta_url = "/fsx/proj -medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
# num_train = 8559 + 300
# num_val = 982
# batch_size = 300

# train_data = wds.WebDataset(train_url, resampled=False)\
#     .decode("torch")\
#     .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
#     .to_tuple("voxels", "images", "coco")\
#     .batched(batch_size, partial=True)

# train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=False)

# train_pred_clips = None
# for train_i, (voxel, img, coco) in enumerate(tqdm(train_dl,total=num_train//batch_size)):
#     with torch.no_grad():
#         voxel=voxel.to(device)
#         for i in range(3):
#             if i==0:
#                 emb_ = voxel2clip(voxel[:,i].float())[:,None]
#             else:
#                 emb_ = torch.cat((emb_, voxel2clip(voxel[:,i].float())[:,None] ),dim=1)
            
#         if train_pred_clips is None:
#             train_pred_clips = emb_
#         else:
#             train_pred_clips = torch.vstack((train_pred_clips, emb_))
            
# np.save('subj01_vitb32image_train_pred_clips.npy', train_pred_clips.detach().cpu().numpy())

# val_data = wds.WebDataset(val_url, resampled=False)\
#     .decode("torch")\
#     .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
#     .to_tuple("voxels", "images", "coco")\
#     .batched(batch_size, partial=True)

# val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# val_pred_clips = None
# for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=num_val//batch_size)):
#     with torch.no_grad():
#         voxel=voxel.to(device)
#         voxel = torch.mean(voxel,axis=1).to(device)
#         emb_ = voxel2clip(voxel.float())
#         if val_pred_clips is None:
#             val_pred_clips = emb_
#         else:
#             val_pred_clips = torch.vstack((val_pred_clips, emb_))
            
# np.save('subj01_vitb32image_test_pred_clips.npy', val_pred_clips.detach().cpu().numpy())


# ## Prep data loader

# In[10]:


batch_size = 300 # same as used in mind_reader

# train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
num_train = 8559 + 300
num_val = 982

val_batch_size = 300
val_loops = 30

voxels_key = 'nsdgeneral.npy' # 1d inputs
# voxels_key = 'wholebrain_3d.npy' #3d inputs

val_data = wds.WebDataset(val_url, resampled=True)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)\
    .with_epoch(val_loops)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break


# ## Forward / Backward quantification

# In[14]:


# clip_extractor = Clipper("ViT-B/32", hidden_state=False, norm_embs=False, device=device)

percent_correct_fwd, percent_correct_bwd = None, None

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=val_loops)):
    with torch.no_grad():
        voxel = torch.mean(voxel,axis=1).to(device) # average across repetitions

        emb = clip_extractor.embed_image(img.to(device)).float() # CLIP-Image
        emb_ = voxel2clip(voxel.float())#.view(len(voxel),-1,768) # CLIP-Brain
        # emb_ = emb_ / torch.norm(emb_[:, 0], dim=-1).reshape(-1, 1, 1)
        
        #emb = clip_extractor.embed_curated_annotations(subj01_annots[trial]) # CLIP-Text
        
        # flatten if necessary
        emb = emb.reshape(len(emb),-1)
        emb_ = emb_.reshape(len(emb_),-1)
        
        # l2norm 
        emb = nn.functional.normalize(emb,dim=-1)
        emb_ = nn.functional.normalize(emb_,dim=-1)
        
        labels = torch.arange(len(emb)).to(device)
        bwd_sim = utils.batchwise_cosine_similarity(emb,emb_)  # clip, brain
        fwd_sim = utils.batchwise_cosine_similarity(emb_,emb)  # brain, clip

        assert len(bwd_sim) == batch_size
        
        if percent_correct_fwd is None:
            cnt=len(fwd_sim)
            percent_correct_fwd = utils.topk(fwd_sim, labels,k=1)
            percent_correct_bwd = utils.topk(bwd_sim, labels,k=1)
        else:
            cnt+=len(fwd_sim)
            percent_correct_fwd += utils.topk(fwd_sim, labels,k=1)
            percent_correct_bwd += utils.topk(bwd_sim, labels,k=1)
print("cnt",cnt,"val_i",val_i)
percent_correct_fwd /= (val_i+1)
percent_correct_bwd /= (val_i+1)
print("fwd percent_correct", percent_correct_fwd)
print("bwd percent_correct", percent_correct_bwd)


# In[ ]:


# fwd percent_correct tensor(0.9369, device='cuda:0')
# bwd percent_correct tensor(0.9011, device='cuda:0')


# In[26]:


percent_correct_fwd, percent_correct_bwd = None, None

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=val_loops)):
    with torch.no_grad():
        voxel = voxel[:,0].to(device) # use first repetition

        emb = clip_extractor.embed_image(img.to(device)) # CLIP-Image
        emb_ = voxel2clip(voxel.float()) # CLIP-Brain
        
        # flatten if necessary
        emb = emb.reshape(len(emb),-1)
        emb_ = emb_.reshape(len(emb_),-1)

        # l2norm 
        emb = nn.functional.normalize(emb,dim=-1)
        emb_ = nn.functional.normalize(emb_,dim=-1)

        labels = torch.arange(len(emb)).to(device)
        bwd_sim = utils.batchwise_cosine_similarity(emb,emb_)  # clip, brain
        fwd_sim = utils.batchwise_cosine_similarity(emb_,emb)  # brain, clip

        assert len(bwd_sim) == batch_size
        
        if percent_correct_fwd is None:
            cnt=len(fwd_sim)
            percent_correct_fwd = utils.topk(fwd_sim, labels,k=1)
            percent_correct_bwd = utils.topk(bwd_sim, labels,k=1)
        else:
            cnt+=len(fwd_sim)
            percent_correct_fwd += utils.topk(fwd_sim, labels,k=1)
            percent_correct_bwd += utils.topk(bwd_sim, labels,k=1)
print("cnt",cnt,"val_i",val_i)
percent_correct_fwd /= (val_i+1)
percent_correct_bwd /= (val_i+1)
print("fwd percent_correct", percent_correct_fwd)
print("bwd percent_correct", percent_correct_bwd)


# ### Brain Diffuser results for comparison

# In[ ]:


sub = 1
test_clips = np.load('brain-diffuser/data/extracted_features/subj{:02d}/nsd_clipvision_test.npy'.format(sub))
pred_clips = np.load('brain-diffuser/data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(sub))


# In[27]:


percent_correct_fwd, percent_correct_bwd = None, None
for loop in tqdm(range(val_loops)):
    with torch.no_grad():
        random300 = np.random.permutation(np.arange(num_val))[:300]
        
        emb = torch.Tensor(test_clips[random300]).to(device)
        emb_ = torch.Tensor(pred_clips[random300]).to(device)
        
        # flatten if necessary
        emb = emb.reshape(len(emb),-1)
        emb_ = emb_.reshape(len(emb_),-1)

        # l2norm 
        emb = nn.functional.normalize(emb,dim=-1)
        emb_ = nn.functional.normalize(emb_,dim=-1)

        labels = torch.arange(len(emb)).to(device)
        bwd_sim = utils.batchwise_cosine_similarity(emb,emb_)  # clip, brain
        fwd_sim = utils.batchwise_cosine_similarity(emb_,emb)  # brain, clip

        assert len(bwd_sim) == batch_size
        
        if percent_correct_fwd is None:
            cnt=len(fwd_sim)
            percent_correct_fwd = utils.topk(fwd_sim, labels,k=1)
            percent_correct_bwd = utils.topk(bwd_sim, labels,k=1)
        else:
            cnt+=len(fwd_sim)
            percent_correct_fwd += utils.topk(fwd_sim, labels,k=1)
            percent_correct_bwd += utils.topk(bwd_sim, labels,k=1)
print("cnt",cnt,"val_i",val_i)
percent_correct_fwd /= (val_i+1)
percent_correct_bwd /= (val_i+1)
print("fwd percent_correct", percent_correct_fwd)
print("bwd percent_correct", percent_correct_bwd)


# ### Plot some of the results

# In[5]:


print("Forward retrieval")
print("Aka given Brain embedding, find correct CLIP embedding")
try:
    fwd_sim = np.array(fwd_sim.cpu())
except:
    fwd_sim = np.array(fwd_sim)
fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(11,12))
for trial in range(4):
    ax[trial, 0].imshow(utils.torch_to_Image(img[trial]))
    ax[trial, 0].set_title("original\nimage")
    ax[trial, 0].axis("off")
    for attempt in range(5):
        which = np.flip(np.argsort(fwd_sim[trial]))[attempt]
        ax[trial, attempt+1].imshow(utils.torch_to_Image(img[which]))
        ax[trial, attempt+1].set_title(f"Top {attempt+1}")
        ax[trial, attempt+1].axis("off")
fig.tight_layout()
plt.show()


# In[6]:


print("Backward retrieval")
print("Aka given CLIP embedding, find correct Brain embedding")
try:
    bwd_sim = np.array(bwd_sim.cpu())
except:
    bwd_sim = np.array(bwd_sim)
fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(11,12))
for trial in range(4):
    ax[trial, 0].imshow(utils.torch_to_Image(img[trial]))
    ax[trial, 0].set_title("original\nimage")
    ax[trial, 0].axis("off")
    for attempt in range(5):
        which = np.flip(np.argsort(bwd_sim[trial]))[attempt]
        ax[trial, attempt+1].imshow(utils.torch_to_Image(img[which]))
        ax[trial, attempt+1].set_title(f"Top {attempt+1}")
        ax[trial, attempt+1].axis("off")
fig.tight_layout()
plt.show()


# # Reconstruction evaluation

# ## Load pretrained models

# In[26]:


# Load image token model
model_name = "v2c_v2" #"pearson4" #"v2c_vers_xmse_1e-3" #"v2c_vers_new_mse5_1e-3" # "v2c_vers"
out_dim = 257 * 768

outdir = f'../train_logs/{model_name}'
ckpt_path = os.path.join(outdir, f'last.pth')
print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("EPOCH: ",checkpoint['epoch'])

with torch.no_grad():
    voxel2clip_img = BrainNetwork(out_dim=out_dim) 
    voxel2clip_img.eval()
    voxel2clip_img.load_state_dict(checkpoint['model_state_dict'], strict=False)
    voxel2clip_img.requires_grad_(False) # dont need to calculate gradients
    voxel2clip_img.to(device)
    pass

del checkpoint


# In[2]:


out_dim = 257 * 768
voxel2clip_kwargs = dict(out_dim=out_dim, norm_type='ln', act_first=False)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

# need folder "checkpoints" with following files
# wget https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json
# wget https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth
out_dim = 768
depth = 6
dim_head = 64
heads = 12 # heads * dim_head = 12 * 64 = 768
timesteps = 100

prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    ).to(device)

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
).to(device)

# model_name = "prior9X"
# outdir = f'../train_logs/{model_name}'
# ckpt_path = os.path.join(outdir, f'last.pth')

model_name = 'prior_nodetr_noncausal_posemb_240_cont'
ckpt_path = '/fsx/proj-medarc/fmri/fMRI-reconstruction-NSD/train_logs/models/prior_nodetr_noncausal_posemb_240_cont/epoch239.pth'

print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
for key in list(state_dict.keys()):
    if 'module.' in key:
        state_dict[key.replace('module.', '')] = state_dict[key]
        del state_dict[key]
diffusion_prior.load_state_dict(state_dict,strict=False)
diffusion_prior.eval().to(device)
diffusion_priors = [diffusion_prior]
pass


# In[4]:


# # need folder "checkpoints" with following files
# # wget https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json
# # wget https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth
# voxel2clip_cls = BrainNetwork(out_dim=768)
# diffusion_prior = BrainDiffusionPrior.from_pretrained(
#     dict(),
#     dict(
#         condition_on_text_encodings=False,
#         timesteps=1000,
#         voxel2clip=voxel2clip_cls,
#     ),
# )

# model_name = "v2c_1x768_1gpu_prior__"
# outdir = f'../train_logs/{model_name}'
# ckpt_path = os.path.join(outdir, f'last.pth')
# print("ckpt_path",ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint['model_state_dict']
# print("EPOCH: ",checkpoint['epoch'])
# for key in list(state_dict.keys()):
#     if 'module.' in key:
#         state_dict[key.replace('module.', '')] = state_dict[key]
#         del state_dict[key]
# diffusion_prior.load_state_dict(state_dict) # messes up expecting voxel2clip, which gets defined below
# diffusion_prior.eval().to(device)
# diffusion_priors = [diffusion_prior]
# pass


# In[5]:


# Load image token model
model_name = "v2c_77x768_1gpu_mse_"#norm_ok"
out_dim = 77 * 768

outdir = f'../train_logs/{model_name}'
ckpt_path = os.path.join(outdir, f'last.pth')
print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location='cpu')
print("EPOCH: ",checkpoint['epoch'])
with torch.no_grad():
    voxel2clip_txt = BrainNetwork(out_dim=out_dim) 
    print("converting ddp model to non-ddp format")
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
            del state_dict[key]
    voxel2clip_txt.eval()
    voxel2clip_txt.load_state_dict(state_dict)
    for param in voxel2clip_txt.parameters():
        param.requires_grad = False # dont need to calculate gradients
    voxel2clip_txt.to(device)
    pass

del checkpoint


# ## Prep data loader

# In[3]:


image_var = 'images'
# train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
num_train = 8559 + 300
num_val = 982

batch_size = val_batch_size = 1

voxels_key = 'nsdgeneral.npy' # 1d inputs
# voxels_key = 'wholebrain_3d.npy' #3d inputs

val_data = wds.WebDataset(val_url, resampled=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break


# ### Load pretrained generative model

# In[14]:


from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
vd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
        # "lambdalabs/sd-image-variations-diffusers",
        vd_cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
    )
vd_pipe.image_unet.eval().to(device)
vd_pipe.vae.eval().to(device)
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")
num_inference_steps = 20

# Set weighting of Dual-Guidance 
text_image_ratio = .0 # .5 means equally weight text and image, 0 means only use image
condition_types = ("text", "image")
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(condition_types):
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer


# In[3]:


# from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, UniPCMultistepScheduler, Transformer2DModel
# from diffusers.models import DualTransformer2DModel
# from diffusers.pipelines.versatile_diffusion.modeling_text_unet import UNetFlatConditionModel

# with torch.no_grad():
#     sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
#     if not os.path.isdir(sd_cache_dir): # download from huggingface if not already downloaded / cached
#         from diffusers import VersatileDiffusionPipeline
#         print("Downloading from huggingface...")
#         sd_pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion")
#         sd_cache_dir = "shi-labs/versatile-diffusion"
#     unet = UNet2DConditionModel.from_pretrained(sd_cache_dir,subfolder="image_unet").to(device)
#     unet.eval() # dont want to train model
#     unet.requires_grad_(False) # dont need to calculate gradients

#     vae = AutoencoderKL.from_pretrained(sd_cache_dir,subfolder="vae").to(device)
#     vae.eval()
#     vae.requires_grad_(False)

#     scheduler = "unipc" # "pndms" or "unipc"

#     text_unet = UNetFlatConditionModel.from_pretrained(sd_cache_dir,subfolder="text_unet").to(device)
#     text_unet.eval() # dont want to train model
#     text_unet.requires_grad_(False) # dont need to calculate gradients

#     noise_scheduler = PNDMScheduler.from_pretrained(sd_cache_dir, subfolder="scheduler")
#     if scheduler == "unipc":
#         noise_scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)
#         num_inference_steps = 20
#     else:
#         num_inference_steps = 50

#     # convert to dual attention         
#     for name, module in unet.named_modules():
#         if isinstance(module, Transformer2DModel):
#             parent_name, index = name.rsplit(".", 1)
#             index = int(index)

#             image_transformer = unet.get_submodule(parent_name)[index]
#             text_transformer = text_unet.get_submodule(parent_name)[index]

#             config = image_transformer.config
#             dual_transformer = DualTransformer2DModel(
#                 num_attention_heads=config.num_attention_heads,
#                 attention_head_dim=config.attention_head_dim,
#                 in_channels=config.in_channels,
#                 num_layers=config.num_layers,
#                 dropout=config.dropout,
#                 norm_num_groups=config.norm_num_groups,
#                 cross_attention_dim=config.cross_attention_dim,
#                 attention_bias=config.attention_bias,
#                 sample_size=config.sample_size,
#                 num_vector_embeds=config.num_vector_embeds,
#                 activation_fn=config.activation_fn,
#                 num_embeds_ada_norm=config.num_embeds_ada_norm,
#             )
#             dual_transformer.transformers[0] = image_transformer
#             dual_transformer.transformers[1] = text_transformer

#             unet.get_submodule(parent_name)[index] = dual_transformer
#             unet.register_to_config(dual_cross_attention=True)
            
#     import logging
#     logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
#     from transformers import CLIPTextModelWithProjection, CLIPTokenizer
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#     text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
#     text_encoder.eval()
#     text_encoder.requires_grad_(False)

#     # annots = np.load("/fsx/proj-medarc/fmri/natural-scenes-dataset/COCO_73k_annots_curated.npy")
    
#     text_image_ratio = .0 # .5 means equally weight text and image, 0 means only use image
#     condition_types = ("text", "image")
#     for name, module in unet.named_modules():
#         if isinstance(module, DualTransformer2DModel):
#             module.mix_ratio = text_image_ratio
#             for i, type in enumerate(condition_types):
#                 if type == "text":
#                     module.condition_lengths[i] = 77
#                     module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
#                 else:
#                     module.condition_lengths[i] = 257
#                     module.transformer_index_for_condition[i] = 0  # use the first (image) transformer


# ## Check autoencoder works

# In[21]:


voxel2sd = Voxel2StableDiffusionModel()
state_dict = torch.load('../train_logs/models/autoencoder/test/ckpt-epoch120.pth', 
                        map_location='cpu')["model_state_dict"]
# for key in list(state_dict.keys()):
#     if 'module.' in key:
#         state_dict[key.replace('module.', '')] = state_dict[key]
#         del state_dict[key]
voxel2sd.load_state_dict(state_dict)
voxel2sd.eval()
voxel2sd.to(device)
pass


# ## Testing Furkan's code

# In[20]:


# for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=num_val)):
#     plt.figure(figsize=(2,2))
#     plt.imshow(utils.torch_to_Image(img))
#     plt.title(f"val_i={val_i}")
#     plt.show()


# ## Reconstruction via diffusion, one at a time
# This will take awhile!!

# In[ ]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
all_images = None
all_clip_recons = None
all_brain_recons = None
all_laion_picks = None

recons_per_clip = 0
recons_per_brain = 8

img2img = False
retrieve = False
plotting = False
v2c_ranking = True

if retrieve: assert batch_size == 1

ind_include = np.arange(num_val) #np.arange(3) #np.arange(num_val)

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=num_val)):
    if val_i<np.min(ind_include):
        continue
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    with torch.no_grad():
        if img2img:
            #ae_preds = voxel2sd(voxel.float())
            #blurry_recons = vae.decode(ae_preds.to(device)/0.18215).sample / 2 + 0.5
            
            blurry_recons = PIL.Image.open(f"blurry_recons/{coco.item()}.png").convert('RGB')
            blurry_recons = transforms.PILToTensor()(blurry_recons)
            blurry_recons = transforms.Resize((512,512))(blurry_recons)
            blurry_recons = (blurry_recons.float() / 255)[None]
        else:
            blurry_recons = None
            
        # prompt = utils.select_annotations(annots[coco.cpu().numpy()], random=True).tolist()
        # prompt = ['High-quality photograph']
        # print(prompt)
        # text_inputs = tokenizer(
        #             prompt,
        #             padding="max_length",
        #             max_length=tokenizer.model_max_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )
        # text_input_ids = text_inputs.input_ids
        # if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        #     attention_mask = text_inputs.attention_mask.to(device)
        # else:
        #     attention_mask = None
        # prompt_embeds = text_encoder(
        #     text_input_ids.to(device),
        #     attention_mask=attention_mask,
        # )
        # embeds = text_encoder.text_projection(prompt_embeds.last_hidden_state)
        # embeds_pooled = prompt_embeds.text_embeds
        # text_token = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)

        # text_token = voxel2clip_txt(voxel.float()).reshape(len(voxel),-1,768)
        # text_token = text_token / torch.norm(text_token[:,0].unsqueeze(1), dim=-1, keepdim=True)
            
        grid, clip_recons, brain_recons, laion_best_picks, recon_img = utils.reconstruct_from_clip(
            img, voxel,
            clip_extractor, vd_pipe.image_unet, vd_pipe.vae, vd_pipe.scheduler,
            voxel2clip_img = voxel2clip_img, 
            diffusion_priors = diffusion_priors,
            text_token = None,
            img_lowlevel = blurry_recons,
            num_inference_steps = 20,
            n_samples_save = batch_size,
            recons_per_clip = recons_per_clip,
            recons_per_brain = recons_per_brain,
            guidance_scale = 3.5,
            img2img_strength = .95, # 0=fully rely on img_lowlevel, 1=not doing img2img
            timesteps_prior = 100,
            seed = seed,
            retrieve = retrieve,
            plotting = plotting,
            v2c_reference = True,
        )
            
        if plotting:
            plt.show()
            # grid.savefig(f'evals/{model_name}_{val_i}.png')
            # plt.close()

        if clip_recons is not None:
            clip_recons = clip_recons[:,0]
        if brain_recons is not None:
            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_clip_recons = clip_recons
            all_images = img
        else:
            if recons_per_brain > 0 or retrieve:
                all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            if recons_per_clip > 0:
                all_clip_recons = torch.vstack((all_clip_recons,clip_recons))
            all_images = torch.vstack((all_images,img))
    if val_i>=np.max(ind_include):
        break

if recons_per_brain > 0 or retrieve:
    if recons_per_brain>0:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
    else:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
if recons_per_clip > 0:
    all_clip_recons = all_clip_recons.view(-1,3,imsize,imsize)

print("all_images.shape",all_images.shape)
if recons_per_brain > 0 or retrieve:
    print("all_brain_recons.shape",all_brain_recons.shape)
if recons_per_clip > 0: 
    print("all_clip_recons.shape",all_clip_recons.shape)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

torch.save(all_brain_recons,f'evals/{model_name}_brain_recons_v2cranking')
# torch.save(all_clip_recons,'evals/all_clip_recons')
# torch.save(all_images,'evals/all_images')


# In[12]:


with torch.no_grad():
    grids,_ = utils.vd_sample_images(
                clip_extractor, diffusion_prior.voxel2clip, vd_pipe, diffusion_prior,
                voxel, img, seed=42,
            )
grids[0]


# In[6]:


from transformers import CLIPVisionModelWithProjection
sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').to(device)
image_encoder.eval()
for param in image_encoder.parameters():
    param.requires_grad = False # dont need to calculate gradients

emb = clip_extractor.embed_image(imgx.to(device)) # Original image
emb = emb.reshape(len(emb),-1)
emb = nn.functional.normalize(emb,dim=-1)


# In[8]:


# clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
# train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
# num_train = 8559 + 300
# batch_size = 100
# train_data = wds.WebDataset(train_url, resampled=False)\
#     .decode("torch")\
#     .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
#     .to_tuple("voxels", "images", "coco")\
#     .batched(batch_size, partial=False)
# train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=False)

# rgb2gray = transforms.Grayscale(num_output_channels=3)
# def mute_image(image_tensor, blend_factor=1.):
#     gray_image = rgb2gray(image_tensor)
#     # Blend the grayscale image with the original image
#     muted_image = (1 - blend_factor) * image_tensor + blend_factor * gray_image
#     return muted_image

# muted_embs = None
# for train_i, (voxel, img, coco) in enumerate(tqdm(train_dl,total=num_train//batch_size)):
#     with torch.no_grad():
#         emb = clip_extractor.embed_image(mute_image(img))
#         if muted_embs is None:
#             muted_embs = emb
#         else:
#             muted_embs = torch.vstack((muted_embs, emb))
# muted_avg = torch.mean(muted_embs,dim=0)[None]


# In[19]:


from diffusers.utils import randn_tensor
guidance_scale = 7.5
torch.manual_seed(0)
do_classifier_free_guidance = guidance_scale > 1.0
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
height = unet.config.sample_size * vae_scale_factor
width = unet.config.sample_size * vae_scale_factor
def decode_latents(latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

preproc = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=224),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

plt.imshow(utils.torch_to_Image(img[[1]]))
plt.show()

with torch.no_grad():
    input_embedding = clip_extractor.embed_image(img[[1]].to(device)).float() + muted_avg
    # input_embedding /= torch.norm(input_embedding[:,-10],dim=-1)
    #input_embedding = voxel2clip(voxel.to(device).float()).reshape(len(voxel),-1,768)
    
    # push = torch.nn.functional.normalize(input_embedding - muted_avg, dim=1)
    # print(push.shape)
    # input_embedding = input_embedding + .9 * push

    input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device)
    print(input_embedding.shape)

    prompt_embeds = torch.randn(1,77,768).to(device)
    prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device)
    input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

    noise_scheduler.set_timesteps(num_inference_steps=20, device=device)
    batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
    shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
    timesteps = noise_scheduler.timesteps
    latents = randn_tensor(shape, device=device, dtype=input_embedding.dtype)
    latents = latents * noise_scheduler.init_noise_sigma
    for i, t in enumerate(tqdm(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    recons = decode_latents(latents).detach().cpu()
utils.torch_to_Image(recons)


# In[25]:


from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, UniPCMultistepScheduler

sd_cache_dir = '/fsx/home-paulscotti/.cache/huggingface/diffusers/models--lambdalabs--sd-image-variations-diffusers/snapshots/a2a13984e57db80adcc9e3f85d568dcccb9b29fc'
if not os.path.isdir(sd_cache_dir): # download from huggingface if not already downloaded / cached
    from diffusers import StableDiffusionImageVariationPipeline
    print("Downloading lambdalabs/sd-image-variations-diffusers from huggingface...")
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="v2.0")
    sd_cache_dir = "lambdalabs/sd-image-variations-diffusers"

unet = UNet2DConditionModel.from_pretrained(sd_cache_dir,subfolder="unet").to(device)
unet.eval() # dont want to train model
unet.requires_grad_(False) # dont need to calculate gradients

vae = AutoencoderKL.from_pretrained(sd_cache_dir,subfolder="vae").to(device)
vae.eval()
vae.requires_grad_(False)

noise_scheduler = PNDMScheduler.from_pretrained(sd_cache_dir, subfolder="scheduler")
noise_scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)
num_inference_steps = 20
    
recons_per_clip = 1
recons_per_brain = 2

print("img-variations reconstructing...")
with torch.no_grad():
    grid, clip_recons, brain_recons, laion_best_picks, recon_img2 = utils.reconstruct_from_clip(
        img, voxel,
        clip_extractor, unet, vae, noise_scheduler,
        voxel2clip_img = voxel2clip, 
        diffusion_priors = diffusion_priors,
        text_token = text_token,
        img_lowlevel = None,#recon_img[None],
        num_inference_steps = num_inference_steps,
        n_samples_save = batch_size,
        recons_per_clip = recons_per_clip,
        recons_per_brain = recons_per_brain,
        guidance_scale = 3.5,
        img2img_strength = .95, # 0=fully rely on img_lowlevel, 1=not doing img2img
        timesteps = 1000,
        seed = seed,
        retrieve = retrieve,
        plotting = plotting,
        variations = True,
    )


# In[24]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
all_images = None
all_clip_recons = None
all_brain_recons = None
all_laion_picks = None

recons_per_clip = 1
recons_per_brain = 1

img2img = True
retrieve = False
plotting = True

if retrieve:
    assert batch_size == 1

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=num_val)):
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            blurry_recons = vae.decode(ae_preds.to(device)/0.18215).sample / 2 + 0.5
        else:
            blurry_recons = None
          
        if versatile or openclip:
            grid, clip_recons, brain_recons, laion_best_picks = utils.reconstruct_from_clip(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                voxel2clip=voxel2clip, 
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_clip = recons_per_clip,
                recons_per_brain = recons_per_brain,
                guidance_scale = 7.5,
                img2img_strength = .8, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps = 1000,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                openclip = openclip,
                versatile = versatile,
                vers_prior_path = None,#'/fsx/proj-medarc/fmri/paulscotti/fMRI-reconstruction-NSD/train_logs',
                num_layers = None,
            )
        else:
            grid, clip_recons, brain_recons, laion_best_picks = utils.reconstruct_from_clip(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                diffusion_priors=diffusion_priors, 
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_clip = recons_per_clip,
                recons_per_brain = recons_per_brain,
                guidance_scale = 7.5,
                img2img_strength = .9, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps = 1000,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                openclip = openclip,
                versatile = versatile,
            )
            
        if plotting:
            grid.savefig(f'evals/{model_name}_{val_i}_{scheduler}.png')
            # plt.close()

        if clip_recons is not None:
            clip_recons = clip_recons[:,0]
        if brain_recons is not None:
            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_clip_recons = clip_recons
            all_images = img
        else:
            if recons_per_brain > 0 or retrieve:
                all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            if recons_per_clip > 0:
                all_clip_recons = torch.vstack((all_clip_recons,clip_recons))
            all_images = torch.vstack((all_images,img))
    if val_i>=4:
        break

if recons_per_brain > 0 or retrieve:
    if recons_per_brain>0:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
    else:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
if recons_per_clip > 0:
    all_clip_recons = all_clip_recons.view(-1,3,imsize,imsize)

print("all_images.shape",all_images.shape)
if recons_per_brain > 0 or retrieve:
    print("all_brain_recons.shape",all_brain_recons.shape)
if recons_per_clip > 0: 
    print("all_clip_recons.shape",all_clip_recons.shape)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# In[10]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
all_images = None
all_clip_recons = None
all_brain_recons = None
all_laion_picks = None

recons_per_clip = 1
recons_per_brain = 1

img2img = False
retrieve = False
plotting = True

if retrieve:
    assert batch_size == 1

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=num_val)):
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            blurry_recons = vae.decode(ae_preds.to(device)/0.18215).sample / 2 + 0.5
        else:
            blurry_recons = None
          
        if versatile or openclip:
            grid, clip_recons, brain_recons, laion_best_picks = utils.reconstruct_from_clip(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                voxel2clip=voxel2clip, 
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_clip = recons_per_clip,
                recons_per_brain = recons_per_brain,
                guidance_scale = 7.5,
                img2img_strength = .9, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps = 1000,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                openclip = openclip,
                versatile = versatile,
            )
        else:
            grid, clip_recons, brain_recons, laion_best_picks = utils.reconstruct_from_clip(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                diffusion_priors=diffusion_priors, 
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_clip = recons_per_clip,
                recons_per_brain = recons_per_brain,
                guidance_scale = 7.5,
                img2img_strength = .9, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps = 1000,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                openclip = openclip,
                versatile = versatile,
            )
            
        if plotting:
            grid.savefig(f'evals/{model_name}_{val_i}_{scheduler}.png')
            # plt.close()

        if clip_recons is not None:
            clip_recons = clip_recons[:,0]
        if brain_recons is not None:
            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_clip_recons = clip_recons
            all_images = img
        else:
            if recons_per_brain > 0 or retrieve:
                all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            if recons_per_clip > 0:
                all_clip_recons = torch.vstack((all_clip_recons,clip_recons))
            all_images = torch.vstack((all_images,img))
    if val_i>=4:
        break

if recons_per_brain > 0 or retrieve:
    if recons_per_brain>0:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
    else:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
if recons_per_clip > 0:
    all_clip_recons = all_clip_recons.view(-1,3,imsize,imsize)

print("all_images.shape",all_images.shape)
if recons_per_brain > 0 or retrieve:
    print("all_brain_recons.shape",all_brain_recons.shape)
if recons_per_clip > 0: 
    print("all_clip_recons.shape",all_clip_recons.shape)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# In[15]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
all_images = None
all_clip_recons = None
all_brain_recons = None
all_laion_picks = None

recons_per_clip = 1
recons_per_brain = 1

img2img = False
retrieve = False
plotting = True

if retrieve:
    assert batch_size == 1

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=num_val)):
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            blurry_recons = vae.decode(ae_preds.to(device)/0.18215).sample / 2 + 0.5
        else:
            blurry_recons = None
          
        if versatile or openclip:
            grid, clip_recons, brain_recons, laion_best_picks = utils.reconstruct_from_clip(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                voxel2clip=voxel2clip, 
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_clip = recons_per_clip,
                recons_per_brain = recons_per_brain,
                guidance_scale = 7.5,
                img2img_strength = .9, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps = 1000,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                openclip = openclip,
                versatile = versatile,
            )
        else:
            grid, clip_recons, brain_recons, laion_best_picks = utils.reconstruct_from_clip(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                diffusion_priors=diffusion_priors, 
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_clip = recons_per_clip,
                recons_per_brain = recons_per_brain,
                guidance_scale = 7.5,
                img2img_strength = .9, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps = 1000,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                openclip = openclip,
                versatile = versatile,
            )
            
        if plotting:
            grid.savefig(f'evals/{model_name}_{val_i}_{scheduler}.png')
            # plt.close()

        if clip_recons is not None:
            clip_recons = clip_recons[:,0]
        if brain_recons is not None:
            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_clip_recons = clip_recons
            all_images = img
        else:
            if recons_per_brain > 0 or retrieve:
                all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            if recons_per_clip > 0:
                all_clip_recons = torch.vstack((all_clip_recons,clip_recons))
            all_images = torch.vstack((all_images,img))
    if val_i>=4:
        break

if recons_per_brain > 0 or retrieve:
    if recons_per_brain>0:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
    else:
        all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
if recons_per_clip > 0:
    all_clip_recons = all_clip_recons.view(-1,3,imsize,imsize)

print("all_images.shape",all_images.shape)
if recons_per_brain > 0 or retrieve:
    print("all_brain_recons.shape",all_brain_recons.shape)
if recons_per_clip > 0: 
    print("all_clip_recons.shape",all_clip_recons.shape)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# In[37]:


mse=nn.MSELoss()
print(mse(eo,ef))
print(eo.shape, ef.shape)
print(eo.min(), ef.min())
print(eo.max(), ef.max())
print(eo.mean(), ef.mean())
print(eo.std(), ef.std())


# In[15]:


img_lowlevel_embeddings = clip_extractor.preprocess(img_lowlevel[[emb_idx]])

img_lowlevel_embeddings = nn.functional.interpolate(img_lowlevel_embeddings, (768,768), mode="area", antialias=False)

init_latents = vae.encode(img_lowlevel_embeddings).latent_dist.sample(generator)
init_latents = vae.config.scaling_factor * init_latents
init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

noise = randn_tensor(shape, generator=generator, device=device)
init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
latents = init_latents


# In[31]:


from diffusers import StableUnCLIPimg2ImgPipeline
with torch.no_grad():
    sd_pipe = StableUnCLIPimg2ImgPipeline.from_pretrained(
        sd_cache_dir, torch_dtype=torch.float16,
    )
    sd_pipe.to(device)

p=sd_pipe(image=utils.torch_to_Image(img),prompt='surfer')
p[0][0]


# In[10]:


from diffusers import StableUnCLIPimg2ImgPipeline
with torch.no_grad():
    sd_pipe = StableUnCLIPimg2ImgPipeline.from_pretrained(
        sd_cache_dir, torch_dtype=torch.float16,
    )
    sd_pipe.to(device)

#     image_embeds = clip_extractor.embed_image(img).float()
#     print(image_embeds.shape)

#     # 8. Encode input prompt
#     prompt_embeds = sd_pipe._encode_prompt(
#         prompt="",
#         device=device,
#         num_images_per_prompt=2,
#         do_classifier_free_guidance=True,
#         negative_prompt="",
#     )
#     print(prompt_embeds.shape)

#     # 9. Prepare image embeddings
#     image_embeds = sd_pipe.noise_image_embeddings(
#         image_embeds=image_embeds,
#         noise_level=0, #0 to 1000
#     )
#     print(image_embeds.shape)
#     noise_pred = unet(
#         latent_model_input,
#         t,
#         encoder_hidden_states=prompt_embeds,
#         class_labels=image_embeds,
#         cross_attention_kwargs=None, 
#     ).sample


# # Loading

# ### concatenate separate recon pipes

# In[2]:


# import torch
# from tqdm import tqdm
# all_brain_recons = None
# for i in tqdm(range(9)):
#     j = i * 109
#     jj = j + 109
#     if jj==981: jj = 982
#     d = torch.load(f'evals/recons_prior_nomixco_86_{j}_{jj}',map_location=torch.device('cpu'))
#     # d = torch.load(f'evals/brain_imgtext_recons_{j}_{jj}',map_location=torch.device('cpu'))
#     # d = torch.load(f'evals/retrieval_recons_{j}_{jj}',map_location=torch.device('cpu'))
#     # e = torch.load(f'evals/retrieval_images_{j}_{jj}',map_location=torch.device('cpu'))

#     if all_brain_recons is None:
#         all_brain_recons = d
#         # all_images = e
#     else:
#         all_brain_recons = torch.vstack((all_brain_recons, d))
#         # all_images = torch.vstack((all_images, e))
# print(all_brain_recons.shape)
# # print(all_images.shape)
# # torch.save(all_images,'evals/all_images')
# torch.save(all_brain_recons,'evals/all_brain_prior_nomixco_86')


# ### load image outputs

# In[6]:


# import torch
# from torchvision import transforms
# import PIL
# from tqdm import tqdm

# toTensor = transforms.ToTensor()
# imgpath = 'brain-diffuser/results/versatile_diffusion/subj01'

# for i in tqdm(range(982)):
#     if i==0:
#         all_brain_recons = toTensor(PIL.Image.open(f'{imgpath}/{i}.png'))[None]
#     else:
#         all_brain_recons = torch.vstack((all_brain_recons, toTensor(PIL.Image.open(f'{imgpath}/{i}.png'))[None]))
# print(all_brain_recons.shape)
# torch.save(all_brain_recons,f'evals/braindiffuser_brain_recons')


# ### load pretrained diffusion image outputs

# In[37]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils
seed=42
utils.seed_everything(seed=seed)

# Load CLIP extractor
# from models import OpenClipper
# clip_extractor = OpenClipper("ViT-H-14", device=device)
# imsize = 768

from models import Clipper
clip_extractor = Clipper("ViT-L/14", hidden_state=False, norm_embs=True, device=device)
imsize = 512

# all_brain_recons = torch.load('evals/all_brain_recons')
# all_brain_recons = torch.load('evals/all_brain_retrievals')
# all_brain_recons = torch.load('evals/all_brain_imgtext_recons')

model_name = 'prior_nodetr_noncausal_posemb_240_cont'
all_brain_recons = torch.load(f'evals/{model_name}_brain_recons_img2img')
# all_brain_recons = torch.load('evals/all_brain_recons')
# all_brain_recons = torch.load('evals/all_blurred_recons')
# all_clip_recons = torch.load('evals/all_clip_recons')
all_images = torch.load('evals/all_images')
# all_laion_picks = torch.load('evals/all_laion_picks')

# all_brain_recons = torch.load(f'evals/braindiffuser_brain_recons')
# all_images = torch.Tensor(np.load('brain-diffuser/data/processed_data/subj01/nsd_test_stim_sub1.npy')/255).permute(0,3,1,2)

print(all_images.shape)
all_images = transforms.Resize((425,425))(all_images)
all_brain_recons = transforms.Resize((425,425))(all_brain_recons)

for ii in range(3):
    plt.imshow(utils.torch_to_Image(all_images[ii]))
    plt.show()
    plt.imshow(utils.torch_to_Image(all_brain_recons[ii]))
    plt.show()


# ## 2-way identification

# In[38]:


def l2norm(x):
    return nn.functional.normalize(x, dim=-1)

def two_way_identification(all_brain_recons, groundtruth, model, preprocess):
    per_correct = []
    l2dist_list = []
    for irecon, recon in enumerate(tqdm(all_brain_recons,total=len(all_brain_recons))):
        with torch.no_grad():   
            if torch.all(recon==0) or torch.all(recon==1):
                print("Completely blank reconstruction?")
                continue
            real = groundtruth[[irecon]]
            fake = model(preprocess(recon)[None]).float().flatten()[None]

            rand = torch.cat((groundtruth[:irecon], groundtruth[irecon:]))
            # rand = model(preprocess(other_recons.to(device))).float()

            l2dist_fake = torch.mean(torch.sqrt((l2norm(real) - l2norm(fake))**2))
            l2dist_rand = torch.mean(torch.sqrt((l2norm(real) - l2norm(rand))**2))
            
            # # cosine similarity is faster and gives same per_correct results
            # l2dist_fake = utils.pairwise_cosine_similarity(real,fake).item()
            # l2dist_rand = utils.pairwise_cosine_similarity(real,rand).item()

            if l2dist_fake < l2dist_rand:
                per_correct.append(1)
            else:
                per_correct.append(0)
            l2dist_list.append(l2dist_fake.detach().cpu().numpy())
    return per_correct, l2dist_list

def two_way_identification_clip(all_brain_recons, groundtruth):
    per_correct = []
    l2dist_list = []
    for irecon, recon in enumerate(tqdm(all_brain_recons,total=len(all_brain_recons))):
        with torch.no_grad():       
            if torch.all(recon==0) or torch.all(recon==1):
                print("Completely blank reconstruction?")
                continue
            real = groundtruth[irecon].unsqueeze(0).to(device)
            fake = clip_extractor.embed_image(recon.unsqueeze(0).to(device)).float()

            rand = torch.cat((groundtruth[:irecon], groundtruth[irecon+1:])).to(device)

            l2dist_fake = torch.mean(torch.sqrt((l2norm(real) - l2norm(fake))**2))
            l2dist_rand = torch.mean(torch.sqrt((l2norm(real) - l2norm(rand))**2))

            # # cosine similarity is faster and gives same per_correct results
            # l2dist_fake = utils.pairwise_cosine_similarity(real,fake).item()
            # l2dist_rand = utils.pairwise_cosine_similarity(real,rand).item()

            if l2dist_fake < l2dist_rand:
                per_correct.append(1)
            else:
                per_correct.append(0)
            l2dist_list.append(l2dist_fake.detach().cpu().numpy())
    return per_correct, l2dist_list


# ## Pre-calculate ground truth image features for all 982 samples

# In[39]:


# all_images = torch.load('evals/all_images')
# alex_early=alex_mid=alex_late=incep=clipimg=torch.Tensor([]).to(device)

# # alexnet #
# from torchvision.models import alexnet, AlexNet_Weights
# alex_weights = AlexNet_Weights.DEFAULT
# alex_model_late = alexnet(weights=alex_weights).eval()
# alex_model_mid = alexnet(weights=alex_weights).eval()
# alex_model_early = alexnet(weights=alex_weights).eval()
# alex_model_late.requires_grad_(False).to(device)
# alex_model_mid.requires_grad_(False).to(device)
# alex_model_early.requires_grad_(False).to(device)
# alex_preprocess = alex_weights.transforms()

# for i,f in enumerate(alex_model_late.features):
#     if i>7:
#         alex_model_late.features[i] = nn.Identity()
#     alex_model_late.avgpool=nn.Identity()
#     alex_model_late.classifier=nn.Identity()
    
# for i,f in enumerate(alex_model_mid.features):
#     if i>4:
#         alex_model_mid.features[i] = nn.Identity()
#     alex_model_mid.avgpool=nn.Identity()
#     alex_model_mid.classifier=nn.Identity()
    
# for i,f in enumerate(alex_model_early.features):
#     if i>1:
#         alex_model_early.features[i] = nn.Identity()
#     alex_model_early.avgpool=nn.Identity()
#     alex_model_early.classifier=nn.Identity()
    
# # inceptionv3 #
# from torchvision.models import inception_v3, Inception_V3_Weights

# incep_weights = Inception_V3_Weights.DEFAULT
# incep_model = inception_v3(weights=incep_weights).eval()
# incep_model.requires_grad_(False).to(device)
# incep_preprocess = incep_weights.transforms()

# incep_model.dropout = nn.Identity()
# incep_model.fc = nn.Identity()

# # clip vit-l/14 #
# clip_extractor = Clipper("ViT-L/14", device=device)

# for ii, orig_image in enumerate(tqdm(all_images,total=len(all_images))):
#     orig_image = orig_image.to(device)
    
#     ## ALEXNET ##
#     alex_early = torch.cat((alex_early,alex_model_early(alex_preprocess(orig_image).unsqueeze(0)).float()))
#     alex_mid = torch.cat((alex_mid,alex_model_mid(alex_preprocess(orig_image).unsqueeze(0)).float()))
#     alex_late = torch.cat((alex_late,alex_model_late(alex_preprocess(orig_image).unsqueeze(0)).float()))
    
#     ## INCEPTION V3 ##
#     incep = torch.cat((incep, incep_model(incep_preprocess(orig_image).unsqueeze(0)).float()))
    
#     ## CLIP ##
#     clipimg = torch.cat((clipimg,clip_extractor.embed_image(orig_image.unsqueeze(0)).float()))
    
# torch.save(alex_early.detach().cpu(),'orig_img_features/alex_early.pt')
# torch.save(alex_mid.detach().cpu(),'orig_img_features/alex_mid.pt')
# torch.save(alex_late.detach().cpu(),'orig_img_features/alex_late.pt')
# torch.save(incep.detach().cpu(),'orig_img_features/incep.pt')
# torch.save(clipimg.detach().cpu(),'orig_img_features/clipimg.pt')


# In[40]:


alex_early=torch.load('orig_img_features/alex_early.pt')
alex_mid=torch.load('orig_img_features/alex_mid.pt')
alex_late=torch.load('orig_img_features/alex_late.pt')
incep=torch.load('orig_img_features/incep.pt')
clipimg=torch.load('orig_img_features/clipimg.pt')


# ### AlexNet

# In[41]:


from torchvision.models import alexnet, AlexNet_Weights
alex_weights = AlexNet_Weights.DEFAULT
alex_model_late = alexnet(weights=alex_weights).eval().to(device)
alex_model_mid = alexnet(weights=alex_weights).eval().to(device)
alex_model_early = alexnet(weights=alex_weights).eval().to(device)
alex_model_late.requires_grad_(False).to(device)
alex_model_mid.requires_grad_(False).to(device)
alex_model_early.requires_grad_(False).to(device)
preprocess = alex_weights.transforms()

for i,f in enumerate(alex_model_late.features):
    if i>7:
        alex_model_late.features[i] = nn.Identity()
    alex_model_late.avgpool=nn.Identity()
    alex_model_late.classifier=nn.Identity()
    
for i,f in enumerate(alex_model_mid.features):
    if i>4:
        alex_model_mid.features[i] = nn.Identity()
    alex_model_mid.avgpool=nn.Identity()
    alex_model_mid.classifier=nn.Identity()
    
for i,f in enumerate(alex_model_early.features):
    if i>1:
        alex_model_early.features[i] = nn.Identity()
    alex_model_early.avgpool=nn.Identity()
    alex_model_early.classifier=nn.Identity()

# layer = 'late, AlexNet(8)'
# print(f"\n---{layer}---")
# all_per_correct, all_l2dist_list = two_way_identification(all_brain_recons.to(device), alex_late.to(device), 
#                                                           alex_model_late.to(device), preprocess)
# print(f"2-way Percent Correct (mu, std): {np.mean(all_per_correct):.4f} | {np.std(all_per_correct):.4f}")
# print(f"Avg l2dist_fake (mu, std): {np.mean(all_l2dist_list):.4f} | {np.std(all_l2dist_list):.4f}")

layer = 'mid, AlexNet(5)'
print(f"\n---{layer}---")
all_per_correct, all_l2dist_list = two_way_identification(all_brain_recons.to(device), alex_mid.to(device), 
                                                          alex_model_mid.to(device), preprocess)
print(f"2-way Percent Correct (mu, std): {np.mean(all_per_correct):.4f} | {np.std(all_per_correct):.4f}")
print(f"Avg l2dist_fake (mu, std): {np.mean(all_l2dist_list):.4f} | {np.std(all_l2dist_list):.4f}")

layer = 'early, AlexNet(2)'
print(f"\n---{layer}---")
all_per_correct, all_l2dist_list = two_way_identification(all_brain_recons.to(device), alex_early.to(device), 
                                                          alex_model_early.to(device), preprocess)
print(f"2-way Percent Correct (mu, std): {np.mean(all_per_correct):.4f} | {np.std(all_per_correct):.4f}")
print(f"Avg l2dist_fake (mu, std): {np.mean(all_l2dist_list):.4f} | {np.std(all_l2dist_list):.4f}")


# ### InceptionV3

# In[42]:


from torchvision.models import inception_v3, Inception_V3_Weights

weights = Inception_V3_Weights.DEFAULT
model = inception_v3(weights=weights).eval()
model.requires_grad_(False).to(device)
preprocess = weights.transforms()

model.dropout = nn.Identity()
model.fc = nn.Identity()
# print(model)

all_per_correct, all_l2dist_list = two_way_identification(all_brain_recons.to(device), incep.to(device), 
                                                          model.to(device), preprocess)
        
print(f"2-way Percent Correct (mu, std): {np.mean(all_per_correct):.4f} | {np.std(all_per_correct):.4f}")
print(f"Avg l2dist_fake (mu, std): {np.mean(all_l2dist_list):.4f} | {np.std(all_l2dist_list):.4f}")


# ### CLIP

# In[15]:


all_per_correct, all_l2dist_list = two_way_identification_clip(all_brain_recons, clipimg)
print(f"2-way Percent Correct (mu, std): {np.mean(all_per_correct):.4f} | {np.std(all_per_correct):.4f}")
print(f"Avg l2dist_fake (mu, std): {np.mean(all_l2dist_list):.4f} | {np.std(all_l2dist_list):.4f}")


# ## SSIM

# In[12]:


# see https://github.com/zijin-gu/meshconv-decoding/issues/3
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

# convert image to grayscale with rgb2grey
img_gray = rgb2gray(all_images.permute((0,2,3,1)))
recon_gray = rgb2gray(all_brain_recons.cpu().permute((0,2,3,1)))

ssim_score = ssim(img_gray, recon_gray, data_range=(img_gray.max()-img_gray.min()), gaussian_weights=True, use_sample_covariance=False)
print(ssim_score)


# ## PixCorr

# In[13]:


# Flatten images while keeping the batch dimension
all_images_flattened = all_images.reshape(len(all_images), -1)
all_brain_recons_flattened = all_brain_recons.view(len(all_brain_recons), -1).cpu()

print(all_images.shape, all_images_flattened.shape)
print(all_brain_recons.shape, all_brain_recons_flattened.shape)

corrsum = 0
for i in tqdm(range(982)):
    corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
corrmean = corrsum / 982

print(corrmean)


# # UMAP

# ## Get CLIP embeddings

# In[15]:


import umap


# In[54]:


clip_extractor = Clipper("ViT-L/14", hidden_state=True, refine=False, norm_embs=True, device=device)

# load annotations for coco73k
annots = np.load("/fsx/proj-medarc/fmri/natural-scenes-dataset/COCO_73k_annots_curated.npy")

batch_size = 1 # increasing it doesnt really speed things up for this 

train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
meta_url = "/fsx/proj -medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
num_train = 8559 + 300
num_val = 982

# train_data = wds.WebDataset(train_url, resampled=False)\
#     .decode("torch")\
#     .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
#     .to_tuple("voxels", "images", "coco")\
#     .batched(batch_size, partial=True)

# train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=False)

train_data = wds.WebDataset(train_url, resampled=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(batch_size, partial=False)

train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=False)

clip_texts = None
clip_images = None
clip_brains = None
clip_brains_aligned = None

generator = torch.Generator(device=device)
generator.manual_seed(seed)

with torch.no_grad():
    for train_i, (voxel, image, coco) in enumerate(tqdm(train_dl,total=num_train)):
        # text = utils.select_annotations(annots[coco], random=False)
        # text_emb = clip_extractor.embed_text(text).float()

        #image_emb = nn.functional.normalize(clip_extractor.embed_image(image).reshape(len(voxel),-1).float(),dim=-1)
        image_emb = clip_extractor.embed_image(image).reshape(len(voxel),-1)

        voxel = torch.mean(voxel,axis=1).to(device) # average across repetitions
        voxel_emb = nn.functional.normalize(voxel2clip(voxel.float()).reshape(len(voxel),257,768)).reshape(len(voxel),-1)
        # voxel_emb = nn.functional.normalize(voxel2clip(voxel.float()).reshape(len(voxel),257,768).reshape(len(voxel),-1),dim=-1)
        
        # voxel_emb = nn.functional.normalize(diffusion_prior.voxel2clip(voxel.float()),dim=-1) * diffusion_prior.image_embed_scale

        # voxel_emb_aligned = diffusion_prior.p_sample_loop(voxel_emb.shape, 
        #                         text_cond = dict(text_embed = voxel_emb), 
        #                         cond_scale = 1., timesteps = 1000, generator=generator,
        #                         )

        if clip_images is None:
            # clip_texts = text_emb
            clip_images = image_emb
            clip_brains = voxel_emb
            # clip_brains_aligned = voxel_emb_aligned
        else:
            # clip_texts = torch.vstack((clip_texts, text_emb))
            clip_images = torch.vstack((clip_images, image_emb))       
            clip_brains = torch.vstack((clip_brains, voxel_emb))       
            # clip_brains_aligned = torch.vstack((clip_brains_aligned, voxel_emb_aligned)) 
        if len(clip_brains) >= 600:
            break
# print("clip_texts.shape",clip_texts.shape)
clip_images = clip_images.reshape(len(clip_images), -1)
clip_brains = clip_brains.reshape(len(clip_brains), -1)
print("clip_images.shape",clip_images.shape)
print("clip_brains.shape",clip_brains.shape)
# print("clip_brains_aligned.shape",clip_brains_aligned.shape)

# np.save("clip_texts",clip_texts.detach().cpu().numpy())
# np.save("clip_images_vers",clip_images.detach().cpu().numpy())
# np.save("clip_brains_vers",clip_brains.detach().cpu().numpy())
# np.save("clip_brains_aligned",clip_brains_aligned.detach().cpu().numpy())

print("Done!")    


# In[55]:


# clip_texts = np.load("clip_texts.npy")
# clip_images = np.load("clip_images_vers.npy")
# clip_brains = np.load("clip_brains_vers.npy")
# clip_brains_aligned = np.load("clip_brains_aligned.npy")

clip_images = clip_images.detach().cpu()
clip_brains = clip_brains.detach().cpu()

# print("clip_texts.shape",clip_texts.shape)
print("clip_images.shape",clip_images.shape)
print("clip_brains.shape",clip_brains.shape)
# print("clip_brains_aligned.shape",clip_brains_aligned.shape)


# In[18]:


combined = np.concatenate((clip_texts,clip_images),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
# colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_text_image.png')
plt.show()


# In[152]:


combined = np.concatenate((clip_texts,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

# colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
colors=np.array([[1,.6,0,.5] for i in range(len(clip_texts))])
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_text_brain.png')
plt.show()


# In[148]:


combined = np.concatenate((clip_images,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_image_brain.png')
plt.show()


# In[27]:


combined = np.concatenate((clip_images,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_image_brain.png')
plt.show()


# In[56]:


# pearson4 (normalized across last dimension of 3d input)
combined = np.concatenate((clip_images,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
# plt.savefig(f'UMAP_image_brain.png')
plt.show()


# In[18]:


# pearsononly_mse
combined = np.concatenate((clip_images,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
# plt.savefig(f'UMAP_image_brain.png')
plt.show()


# In[11]:


# pearsononly_
combined = np.concatenate((clip_images,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
# plt.savefig(f'UMAP_image_brain.png')
plt.show()


# In[11]:


# mse5 finetuned more
combined = np.concatenate((clip_images,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_image_brain.png')
plt.show()


# In[149]:


combined = np.concatenate((clip_images,clip_brains_aligned),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
# colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[1,0,.6,.5] for i in range(len(clip_brains_aligned))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_image_brain-aligned.png')
plt.show()


# In[150]:


combined = np.concatenate((clip_brains,clip_brains_aligned),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,1,0,.5] for i in range(len(clip_brains))])
colors=np.concatenate((colors, np.array([[1,0,.6,.5] for i in range(len(clip_brains_aligned))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_brain_brain-aligned.png')
plt.show()


# In[13]:


combined = np.concatenate((clip_images,clip_brains,clip_brains_aligned),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))
colors=np.concatenate((colors, np.array([[1,0,.6,.5] for i in range(len(clip_brains_aligned))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_image_brain_brain-aligned.png')
plt.show()


# In[ ]:


combined = np.concatenate((clip_images,clip_texts,clip_brains),axis=0)
print("combined.shape",combined.shape)

reducer = umap.UMAP()
embedding = reducer.fit_transform(combined)

colors=np.array([[0,0,1,.5] for i in range(len(clip_images))])
colors=np.concatenate((colors, np.array([[1,.6,0,.5] for i in range(len(clip_texts))])))
colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_brains))])))

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors)
plt.title('UMAP projection',fontsize=16)
plt.savefig(f'UMAP_text_image_brain.png')
plt.show()

