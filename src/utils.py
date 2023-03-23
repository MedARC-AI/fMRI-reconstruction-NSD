import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import webdataset as wds
import tempfile
from torchvision.utils import make_grid
from diffusers.utils import randn_tensor
from models import Clipper
import json
from torchmetrics.image.fid import FrechetInceptionDistance

from PIL import Image, UnidentifiedImageError
import traceback
import requests
import time
import io
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def gather_features(image_features, voxel_features, accelerator):  
    all_image_features = accelerator.gather(image_features)
    if voxel_features is not None:
        all_voxel_features = accelerator.gather(voxel_features)
        return all_image_features, all_voxel_features
    return all_image_features

def soft_clip_loss(preds, targs, temp=0.125, distributed=False, accelerator=None):
    if not distributed:
        clip_clip = (targs @ targs.T)/temp
        brain_clip = (preds @ targs.T)/temp
    else:
        all_targs = gather_features(targs, None, accelerator)
        clip_clip = (targs @ all_targs.T)/temp
        brain_clip = (preds @ all_targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.T.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0]).to(voxels.device)
    voxels_shuffle = voxels[perm]
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

<<<<<<< HEAD
def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, accelerator=None, local_rank=None):
=======
def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, local_rank=None):
>>>>>>> e1f44fb0484c63e3ea9e5538ca2ef407a965e9ac
    if distributed:
        all_targs = gather_features(targs, None, accelerator)
        brain_clip = (preds @ all_targs.T)/temp
    else:
        brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        if distributed:
            probs_all = torch.zeros_like(brain_clip)
            probs_all[:, local_rank*brain_clip.shape[0]:(local_rank+1)*brain_clip.shape[0]] = probs
            probs = probs_all

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
<<<<<<< HEAD
        #print('mixco loss: ', loss.item())
=======
        # print('mixco loss: ', loss.item())
>>>>>>> e1f44fb0484c63e3ea9e5538ca2ef407a965e9ac
        return loss
    else:
        return F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))

def dcl(preds, targs, temp=0.1):
    # adapted from https://github.com/raminnakhli/Decoupled-Contrastive-Learning/blob/main/loss/dcl.py
    clip_clip = (targs @ targs.T)
    brain_clip = (preds @ targs.T)
    
    positive_loss = -torch.diag(brain_clip) / temp
    
    neg_similarity = torch.cat((clip_clip, brain_clip), dim=1) / temp
    neg_mask = torch.eye(preds.size(0), device=preds.device).repeat(1, 2)
    negative_loss = torch.logsumexp(neg_similarity + neg_mask, dim=1, keepdim=False)
    return (positive_loss + negative_loss).mean()

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def plot_brainnet(train_losses, train_fwd_topk, train_bwd_topk, val_losses, val_fwd_topk, val_bwd_topk, lrs):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, figsize=(23,3))
    ax1.set_title(f"Training Loss\n(final={train_losses[-1]:.3f})")
    ax1.plot(train_losses)
    ax2.set_title(f"Training Top-1 (fwd)\n(final={train_fwd_topk[-1]:.3f})")
    ax2.plot(train_fwd_topk)
    ax3.set_title(f"Training Top-1 (bwd)\n(final={train_bwd_topk[-1]:.3f})")
    ax3.plot(train_bwd_topk)
    ax4.set_title(f"Val Loss\n(final={val_losses[-1]:.3f})")
    ax4.plot(val_losses)
    ax5.set_title(f"Val Top-1 (fwd)\n(final={val_fwd_topk[-1]:.3f})")
    ax5.plot(val_fwd_topk)
    ax6.set_title(f"Val Top-1 (bwd)\n(final={val_bwd_topk[-1]:.3f})")
    ax6.plot(val_bwd_topk)
    ax7.set_title(f"Learning Rate")
    ax7.plot(lrs)
    fig.tight_layout()
    #fig.suptitle('BrainNet')
    plt.show()

def plot_brainnet_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_losses=checkpoint['train_losses']
    train_fwd_topk=checkpoint['train_fwd_topk']
    train_bwd_topk=checkpoint['train_bwd_topk']
    val_losses=checkpoint['val_losses']
    val_fwd_topk=checkpoint['val_fwd_topk']
    val_bwd_topk=checkpoint['val_bwd_topk']
    lrs=checkpoint['lrs']
    plot_brainnet(train_losses, train_fwd_topk, train_bwd_topk, val_losses, val_fwd_topk, val_bwd_topk, lrs)

def plot_prior(losses, val_losses, lrs, sims, val_sims):
    # rolling over epoch
    # losses_ep = pd.Series(losses).rolling(int(np.ceil(24983/batch_size))).mean().values
    # val_losses_ep = pd.Series(val_losses).rolling(int(np.ceil(492/batch_size))).mean().values
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 4))
    ax1.set_title(f"Training Loss\n(final={losses[-1]:.3f})")
    ax1.plot(losses)
    # ax1.plot(losses_ep)
    ax2.set_title(f"Val Loss\n(final={val_losses[-1]:.3f})")
    ax2.plot(val_losses)
    # ax2.plot(val_losses_ep)
    ax3.set_title(f"Learning Rate")
    ax3.plot(lrs)
    ax4.set_title(f"Training sims\n(final={sims[-1]:.3f})")
    ax4.plot(sims)
    ax5.set_title(f"Val sims\n(final={val_sims[-1]:.3f})")
    ax5.plot(val_sims)
    fig.tight_layout()
    plt.show()
    
def plot_prior_ckpt(ckpt_path):
    prior_checkpoint = torch.load(ckpt_path, map_location=device)
    losses = prior_checkpoint['train_losses']
    val_losses = prior_checkpoint['val_losses']
    lrs = prior_checkpoint.get('lrs', [3e-4]*len(losses))
    sims = prior_checkpoint.get('sims', [0.0]*len(losses))
    val_sims = prior_checkpoint.get('val_sims', [0.0]*len(val_losses))
    plot_prior(losses, val_losses, lrs, sims, val_sims)

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_huggingface_urls(commit='9947586218b6b7c8cab804009ddca5045249a38d'):
    """
    You can use commit='main' is the most up to date data.
    Before the new data was added is commit "9947586218b6b7c8cab804009ddca5045249a38d".
    """
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/"
    train_url = base_url + commit + "/webdataset/train/train_subj01_{0..49}.tar"
    val_url = base_url + commit + "/webdataset/val/val_subj01_0.tar"
    return train_url, val_url

def split_by_node(urls):
    node_id, node_count = accelerator.state.local_process_index, accelerator.state.num_processes
    return urls[node_id::node_count]

def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id::wi.num_workers]
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def get_dataloaders(
    batch_size,
    image_var='images',
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/tmp/wds-cache",
    n_cache_recs=0,
    seed=0,
    voxels_key="nsdgeneral.npy",
    val_batch_size=None,
<<<<<<< HEAD
    to_tuple=["voxels", "images", "trial"],
    local_rank=0,
=======
    to_tuple=["voxels", "images", "__key__"], # include trial with ["voxels", "images", "trial", "__key__"]
    test_is_val=False,
>>>>>>> e1f44fb0484c63e3ea9e5538ca2ef407a965e9ac
):
    if local_rank==0: print("Getting dataloaders...")
    assert image_var == 'images'

    train_url_hf, val_url_hf = get_huggingface_urls()
    # default to huggingface urls if not specified
    if train_url is None:
        train_url = train_url_hf
    if val_url is None:
        val_url = val_url_hf

    if num_devices is None:
        num_devices = torch.cuda.device_count()
    
    if num_workers is None:
        num_workers = num_devices
    
    if meta_url is None:
        # for commits up to 9947586218b6b7c8cab804009ddca5045249a38d
<<<<<<< HEAD
        if num_train is None:
            num_train = 24983
        if num_val is None:
            num_val = 492
    else:
        metadata = json.load(open(meta_url))
        if num_train is None:
            num_train = metadata['totals']['train']
        if num_val is None:
            num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size
=======
        num_train = 24983
        num_val = 492
        assert test_is_val == False, "test_is_val == True not supported without a meta_url"
    else:
        metadata = json.load(open(meta_url))
        if test_is_val:
            # `train_url` contains train + val and `val_url` contains test
            num_train = metadata['totals']['train'] + metadata['totals']['val']
            num_val = metadata['totals']['test']
        else:
            num_train = metadata['totals']['train']
            num_val = metadata['totals']['val']
>>>>>>> e1f44fb0484c63e3ea9e5538ca2ef407a965e9ac
        
    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    if local_rank==0: print("\nnum_train",num_train)
    if local_rank==0: print("global_batch_size",global_batch_size)
    if local_rank==0: print("batch_size",batch_size)
    if local_rank==0: print("num_workers",num_workers)
    if local_rank==0: print("num_batches",num_batches)
    if local_rank==0: print("num_worker_batches", num_worker_batches)
    
    if 'http' not in train_url:
        # don't use cache if train_url is for local path
        cache_dir = None
    print("cache_dir", cache_dir)
    if cache_dir is not None and not os.path.exists(cache_dir):
        os.makedirs(cache_dir,exist_ok=True)
    
    train_data = wds.WebDataset(train_url, resampled=True, cache_dir=cache_dir, nodesplitter=wds.split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(batch_size, partial=True)\
        .with_epoch(num_worker_batches)

    if n_cache_recs > 0:
        train_data = train_data.compose(wds.DBCache, os.path.join(cache_dir, "cache-train.db"),  n_cache_recs)
        
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=None, num_workers=num_workers, shuffle=False)

    # Validation
    # just using the first 300 samples! no shuffling!
    val_num_workers = num_workers
    
    if local_rank==0: print("\nnum_val",num_val)
    if local_rank==0: print("val_batch_size",val_batch_size)
    if local_rank==0: print("val_num_workers",val_num_workers)
    
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=wds.split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(val_batch_size, partial=False)
    
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=val_num_workers, shuffle=False)

    if n_cache_recs > 0:
        val_data = val_data.compose(wds.DBCache, os.path.join(cache_dir, "cache-val.db"),  n_cache_recs)

    return train_dl, val_dl, num_train, num_val

@torch.no_grad()
def sample_images(
    clip_extractor, brain_net, sd_pipe, diffusion_prior, voxel, img_input, 
    annotations=None,
    num_inference_steps=50,
    clip_guidance_scale=7.5,
    vox_guidance_scale=7.5,
    num_per_sample=4,
    prior_timesteps=None,
    seed=None,
    verbose=True,
    device='cuda',
):

    def null_sync(t, *args, **kwargs):
        return [t]

    def convert_imgs_for_fid(imgs):
        # Convert from [0, 1] to [0, 255] and from torch.float to torch.uint8
        return imgs.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

    fid = FrechetInceptionDistance(feature=64, dist_sync_fn=null_sync).to(device)

    # inside FID it will resize to 300x300 from 256x256
    # [n, 3, 256, 256]
    # print('img_input.shape', img_input.shape)
    fid.update(convert_imgs_for_fid(img_input.to(device)), real=True)
    
    assert voxel.shape[0] == img_input.shape[0], 'batch dim must be the same for voxels and images'
    n_examples = voxel.shape[0]

    clip_extractor.eval()
    brain_net.eval()
    diffusion_prior.eval()

    if seed is not None:
        # set seed
        g_cuda = torch.Generator(device=device)
        g_cuda.manual_seed(seed)

    # for brain guided images (specific to 512 x 512 generation size)
    latents = torch.randn([num_per_sample, 4, 64, 64], device=device, generator=g_cuda)
    
    # use the same latent as the first brain guided image for max similarity
    # clip_latents = torch.randn([1, 4, 64, 64], device=device, generator=g_cuda)
    clip_latents = latents[0].unsqueeze(0).clone()

    grids = []

    for idx in range(n_examples):
        print('sampling for image', idx+1, 'of', n_examples, flush=True)

        img_orig = img_input[[idx]]
        image = clip_extractor.resize_image(img_orig)

        # Original clip embedding:
        if annotations is None:
            clip_emb = clip_extractor.embed_image(image)
        else:
            print('Sampling with CLIP text guidance')
            # random=False will use the first prompt here, which could be different from training 
            # but should be the same during validation
            annots = select_annotations(annotations[[idx]], random=False)
            clip_emb = clip_extractor.embed_text(annots)

        # clip_emb = sd_pipe._encode_image(tform(image), device, 1, False).squeeze(1)
        norm_orig = clip_emb.norm().item()

        # Encode voxels to CLIP space
        image_embeddings = brain_net(voxel[[idx]].to(device).float())
        norm_pre_prior = image_embeddings.norm().item()
        
        # image_embeddings = nn.functional.normalize(image_embeddings, dim=-1) 
        # image_embeddings *= clip_emb[1].norm()/image_embeddings.norm() # note: this is cheating to equate norm scaling

        image_embeddings = diffusion_prior.p_sample_loop(image_embeddings.shape, 
                                            text_cond = dict(text_embed = image_embeddings), 
                                            cond_scale = 1., timesteps = prior_timesteps,
                                            generator=g_cuda
                                            )
        norm_post_prior = image_embeddings.norm().item()

        if verbose:
            cos_sim = nn.functional.cosine_similarity(image_embeddings, clip_emb, dim=1).item()
            mse = nn.functional.mse_loss(image_embeddings, clip_emb).item()
            print(f"cosine sim: {cos_sim:.3f}, MSE: {mse:.5f}, norm_orig: {norm_orig:.3f}, "
                  f"norm_pre_prior: {norm_pre_prior:.3f}, norm_post_prior: {norm_post_prior:.3f}", flush=True)

        # duplicate the embedding to serve classifier free guidance
        image_embeddings = image_embeddings.repeat(num_per_sample, 1)
        image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings]).unsqueeze(1).to(device)

        # duplicate the embedding to serve classifier free guidance
        clip_emb = torch.cat([torch.zeros_like(clip_emb), clip_emb]).unsqueeze(1).to(device).float()        

        # TODO: passing sizes doesn't seem to work, so we're using None for now
        # width, height = 256, 256
        width, height = None, None

        with torch.inference_mode(), torch.autocast(device):
            # [1, 3, 512, 512]
            img_clip = sd_pipe(
                image_embeddings=clip_emb,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                guidance_scale=clip_guidance_scale, 
                latents=clip_latents,
                width=width,
                height=height,
                generator=g_cuda,
            )

            # [4, 3, 512, 512]
            imgs_brain = sd_pipe(
                image_embeddings=image_embeddings,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_per_sample,
                guidance_scale=vox_guidance_scale,
                latents=latents,
                width=width,
                height=height,
                generator=g_cuda,
            )

            # print('img_clip.shape', img_clip.shape)
            # print('imgs_brain.shape', imgs_brain.shape)

        # inside FID it will resize to 300x300 from 512x512
        print('Done sampling images, updating FID', flush=True)
        fid.update(convert_imgs_for_fid(imgs_brain.to(device)), real=False)
        print('Done updating FID', flush=True)
        
        # resizing for now since passing target sizes into sd_pipe doesn't work
        size = (256, 256)
        img_clip = nn.functional.interpolate(img_clip, size, mode="area", antialias=False)
        imgs_brain = nn.functional.interpolate(imgs_brain, size, mode="area", antialias=False)
        
        imgs_all = torch.cat((img_orig.to(device), img_clip, imgs_brain), 0)
        grid = torch_to_Image(
            make_grid(imgs_all, nrow=2+4, padding=10).detach()
        )
        grids.append(grid)

    return grids, fid

def save_ckpt(model, optimizer, losses, val_losses, lrs, epoch, tag, outdir):
        ckpt_path = os.path.join(outdir, f'ckpt-{tag}.pth')
        print(f'saving {ckpt_path}')
        state_dict = model.state_dict()
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

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))


size = (336, 336)  #  (1920//4, 1080//4)


def pad_resize(im, size, fit=True):
    ratio = (max if fit else min)((w / s for s, w in zip(im.size, size)))
    im = im.resize([int(s * ratio) for s, w in zip(im.size, size)])
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(im, [w // 2 - s // 2 for s, w in zip(im.size, size)])
    return canvas

def crop_resize(im, size):
    assert size[0] == size[1]  # unnecessary, bad design, why
    w = size[0]
    im = im.convert("RGB")
    a, b = im.size
    im = im.resize((w if a < b else int(a / b * w), w if b <= a else int(b / a * w)))
    a, b = im.size
    x, y = (a - w) // 2, (b - w) // 2
    im = im.crop((x, y, x + w, y + w))
    im = im.resize((w, w))
    return im


def img_result(result, size=size):
    try:
        return crop_resize(  # pad_resize(
            Image.open(
            io.BytesIO(
                requests.get(result["url"]).content  # , stream=True).raw
            )
        ).convert("RGB"), size)  # .resize(size).convert("RGB")
    except (TypeError, requests.ConnectionError, UnidentifiedImageError):
        #traceback.print_exc()
        return None  # Image.new("RGB", size)


def query_laion(text=None, emb=None, num=20):
    # Soft requirement
    try:
        from clip_retrieval.clip_client import ClipClient, Modality
    except ImportError:
        raise ValueError("Install clip-retrieval for retrieval support")
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        aesthetic_score=9,
        aesthetic_weight=0.15,
        modality=Modality.IMAGE,
        num_images=max(4, num * 2),
        use_violence_detector=False,
        use_safety_model=False
    )

    while True:
        try:
            while len(emb.shape) > 1:
                emb = emb[0]
            results = client.query(text=text, embedding_input=(emb / np.linalg.norm(emb)).tolist() if emb is not None else None)
        except Exception:
            traceback.print_exc()
            time.sleep(1)
            continue
        break
    if not results:
        return Image.new("RGB", size)
    imgs = [img_result(result) for result in results]
    imgs = [im for im in imgs if im is not None][:num]
    return imgs


# below is alternative to sample_images that can also handle img2img reference
@torch.no_grad()
def reconstruct_from_clip(
    image, voxel,
    diffusion_priors,
    clip_extractor,
    unet, vae, noise_scheduler,
    img_lowlevel = None,
    num_inference_steps = 50,
    n_samples_save = 4,
    recons_per_clip = 2,
    recons_per_brain = 4,
    guidance_scale = 7.5,
    img2img_strength = .6,
    timesteps = 1000,
    seed = 0,
    distributed = False,
    retrieve=False,
):
    def decode_latents(latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    if not isinstance(diffusion_priors, list):
        diffusion_priors = [diffusion_priors]
    
    voxel=voxel[:n_samples_save]
    image=image[:n_samples_save]

    do_classifier_free_guidance = guidance_scale > 1.0
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = unet.config.sample_size * vae_scale_factor
    width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Prep CLIP-Image embeddings for original image for comparison with reconstructions
    clip_embeddings = clip_extractor.embed_image(image).float()

    # Encode voxels to CLIP space
    brain_clip_embeddings_sum = None
    for diffusion_prior in diffusion_priors:
        if distributed:
            diffusion_prior.module.voxel2clip.eval()
            brain_clip_embeddings0 = diffusion_prior.module.voxel2clip(voxel.to(device).float())
            # NOTE: requires fork of DALLE-pytorch for generator arg
            brain_clip_embeddings = diffusion_prior.module.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps, #1000 timesteps used from nousr pretraining
                                                generator=generator
                                                )
        else:
            diffusion_prior.voxel2clip.eval()
            brain_clip_embeddings0 = diffusion_prior.voxel2clip(voxel.to(device).float())
            # NOTE: requires fork of DALLE-pytorch for generator arg
            brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timesteps, #1000 timesteps used from nousr pretraining
                                                generator=generator
                                                )
        if brain_clip_embeddings_sum is None:
            brain_clip_embeddings_sum = brain_clip_embeddings
        else:
            brain_clip_embeddings_sum += brain_clip_embeddings

    # average embeddings for all diffusion priors
    brain_clip_embeddings = brain_clip_embeddings_sum / len(diffusion_priors)

    # Now enter individual image processing loop
    clip_recons = None
    brain_recons = None
    img2img_refs = None
    for e, emb in enumerate([clip_embeddings, brain_clip_embeddings]):
        if e==0:
            embed_type = 'clip'
        else:
            embed_type = 'brain'
        for emb_idx, input_embedding in enumerate(emb):
            if embed_type == 'clip':
                recons_per_sample = recons_per_clip
            else:
                recons_per_sample = recons_per_brain
            if recons_per_sample == 0: continue
            if embed_type == "brain" and retrieve:
                try:
                    image_retrieved0 = query_laion(emb=brain_clip_embeddings0.detach().cpu().numpy().flatten())
                    image_retrieved = (Image_to_torch(image_retrieved0[0])[0].unsqueeze(0) + 1) / 2
                    retrieved_clip = clip_extractor.embed_image(image_retrieved).float()
                    
                    # crude way to elminate coco image being the nearest neighbor
                    if (torch.abs(torch.mean(clip_embeddings - retrieved_clip)) < .0045):
                        image_retrieved = (Image_to_torch(image_retrieved0[1])[0].unsqueeze(0) + 1) / 2
                        retrieved_clip = clip_extractor.embed_image(image_retrieved).float()
                except:
                    print("query_laion failed! probably due to 'TooManyRedirects: Exceeded 30 redirects.' error.")
                    image_retrieved = torch.zeros_like(brain_clip_embeddings0)
                    retrieved_clip = image_retrieved
                
            input_embedding = input_embedding.repeat(recons_per_sample, 1)
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).unsqueeze(1).to(device)
            
            # 4. Prepare timesteps
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            
            # 5b. Prepare latent variables
            batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
            shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
            if (embed_type == "brain") and (img_lowlevel is not None): # use img_lowlevel for img2img initialization
                img_lowlevel = img_lowlevel[:n_samples_save]
                # img_lowlevel = transforms.functional.gaussian_blur(img_lowlevel,kernel_size=41)
                # img_lowlevel = image_retrieved.float()
                # img_lowlevel = nn.functional.interpolate(img_lowlevel, 512, mode="area", antialias=False).to(device)
                
                init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
                t_start = max(num_inference_steps - init_timestep, 0)
                timesteps = noise_scheduler.timesteps[t_start:]
                latent_timestep = timesteps[:1].repeat(batch_size)

                if img2img_refs is None:
                    img2img_refs = img_lowlevel[[emb_idx]]
                elif img2img_refs.shape[0] <= emb_idx:
                    img2img_refs = torch.cat((img2img_refs, img_lowlevel[[emb_idx]]))
                img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel[[emb_idx]])

                init_latents = vae.encode(img_lowlevel_embeddings).latent_dist.sample(generator)
                init_latents = vae.config.scaling_factor * init_latents
                init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

                noise = randn_tensor(shape, generator=generator, device=device)
                init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
                latents = init_latents
            else:
                timesteps = noise_scheduler.timesteps
                latents = randn_tensor(shape, generator=generator, device=device, dtype=input_embedding.dtype)
                latents = latents * noise_scheduler.init_noise_sigma

            # 7. Denoising loop
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            recons = decode_latents(latents).detach().cpu()

            if embed_type == 'clip':
                if clip_recons is None:
                    clip_recons = recons.unsqueeze(0)
                else:
                    clip_recons = torch.cat((clip_recons,recons.unsqueeze(0)))
            elif embed_type == 'brain':
                if brain_recons is None:
                    brain_recons = recons.unsqueeze(0)
                else:
                    brain_recons = torch.cat((brain_recons,recons.unsqueeze(0)))

    # compare CLIP embedding of LAION nearest neighbor to your brain reconstructions
    best_picks = np.zeros(n_samples_save)
    if embed_type == "brain" and retrieve:
        for im in range(n_samples_save):
            brain_clips = clip_extractor.embed_image(brain_recons[im])
            cos_sims_to_neighbor = batchwise_cosine_similarity(brain_clips.float(), retrieved_clip)
            best_picks[im] = int(torch.argmax(cos_sims_to_neighbor))
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    laion_samples = 1 if retrieve else 0
    num_xaxis_subplots = 1+img2img_samples+laion_samples+recons_per_clip+recons_per_brain
    fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*3,8*n_samples_save),
                           facecolor=(1, 1, 1))
    if n_samples_save > 1:
        for im in range(n_samples_save):
            ax[im][0].set_title(f"Original Image")
            ax[im][0].imshow(torch_to_Image(image[im]))
            if img2img_samples == 1:
                ax[im][1].set_title(f"Img2img ({img2img_strength})")
                ax[im][1].imshow(torch_to_Image(img2img_refs[im]))
            for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_clip-recons_per_brain,num_xaxis_subplots-laion_samples-recons_per_brain)):
                recon = clip_recons[im][ii]
                ax[im][i].set_title(f"Recon {ii+1} from orig CLIP")
                ax[im][i].imshow(torch_to_Image(recon))
            for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_brain,num_xaxis_subplots-laion_samples)):
                recon = brain_recons[im][ii]
                if ii == best_picks[im]:
                    ax[im][i].set_title(f"Best Reconstruction",fontweight='bold')
                else:
                    ax[im][i].set_title(f"Recon {ii+1} from brain")
                ax[im][i].imshow(torch_to_Image(recon))
            if retrieve:
                ax[im][-1].set_title(f"LAION5b top neighbor")
                ax[im][-1].imshow(torch_to_Image(image_retrieved))
            for i in range(num_xaxis_subplots):
                ax[im][i].axis('off')
    else:   
        im = 0
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img2img_refs[im]))
        for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_clip-recons_per_brain,num_xaxis_subplots-recons_per_brain-laion_samples)):
            recon = clip_recons[im][ii]
            ax[i].set_title(f"Recon {ii+1} from orig CLIP")
            ax[i].imshow(torch_to_Image(recon))
        for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_brain,num_xaxis_subplots-laion_samples)):
            recon = brain_recons[im][ii]
            if ii == best_picks[im]:
                ax[i].set_title(f"Best Reconstruction",fontweight='bold')
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
        if retrieve:
            ax[-1].set_title(f"LAION5b top neighbor")
            ax[-1].imshow(torch_to_Image(image_retrieved))
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    
    return fig, clip_recons, brain_recons, best_picks

def save_augmented_images(imgs, keys, path):
    """
    For saving augmented images generated with SD image variation pipeline.
    """
    assert imgs.ndim == 4
    # batch, channel, height, width
    assert imgs.shape[0] == len(keys)

    to_pil = transforms.ToPILImage()    

    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = to_pil(img)

        # make a directory for each key
        key_dir = os.path.join(path, keys[i])
        os.makedirs(key_dir, exist_ok=True)
        
        # count the number of images in the directory
        count = len(glob(key_dir + '/*.jpg'))
        
        # save with an incremented count
        img.save(os.path.join(key_dir, '%04d.jpg' % (count + 1)))

def select_annotations(annots, random=False):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[0, rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[0, j] != '':
                    t = b[0, j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt