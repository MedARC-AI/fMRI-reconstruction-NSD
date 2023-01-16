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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_augment = transforms.Compose([
                transforms.RandomCrop(size=(140,140)),
                transforms.RandomHorizontalFlip(p=.5),
                transforms.ColorJitter(.4,.4,.2,.1),
                transforms.RandomGrayscale(p=.2),
            ])

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    return (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5

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

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.T.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

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

def plot_prior(losses, val_losses, lrs):
    # rolling over epoch
    # losses_ep = pd.Series(losses).rolling(int(np.ceil(24983/batch_size))).mean().values
    # val_losses_ep = pd.Series(val_losses).rolling(int(np.ceil(492/batch_size))).mean().values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    ax1.set_title(f"Training Loss\n(final={losses[-1]:.3f})")
    ax1.plot(losses)
    # ax1.plot(losses_ep)
    ax2.set_title(f"Val Loss\n(final={val_losses[-1]:.3f})")
    ax2.plot(val_losses)
    # ax2.plot(val_losses_ep)
    ax3.set_title(f"Learning Rate")
    ax3.plot(lrs)
    fig.tight_layout()
    #fig.suptitle('DiffusionPrior')
    plt.show()
    
def plot_prior_ckpt(ckpt_path, max_steps=None):
    prior_checkpoint = torch.load(ckpt_path, map_location=device)
    losses = prior_checkpoint['train_losses']
    val_losses = prior_checkpoint['val_losses']
    lrs = prior_checkpoint.get('lrs', [3e-4]*len(losses))
    if max_steps is not None:
        losses = losses[:max_steps]
        val_losses = val_losses[:max_steps]
        lrs = lrs[:max_steps]
    plot_prior(losses, val_losses, lrs)

def get_dataloaders(
    batch_size,
    image_var,
    num_devices=None,
    num_workers=None,
    train_url="/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_split/train/train_subj01_{0..49}.tar",
    val_url="/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_split/val/val_subj01_0.tar",
):
    print("batch_size",batch_size)
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    print("num_devices",num_devices)
    if num_workers is None:
        num_workers = num_devices
    print("num_workers",num_workers)
    num_samples = 24983 # see metadata.json in webdataset_split folder
    global_batch_size = batch_size * num_devices
    print("global_batch_size",global_batch_size)
    num_batches = math.floor(num_samples / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    print("num_worker_batches",num_worker_batches)

    train_data = wds.DataPipeline([wds.ResampledShards(train_url),
                        wds.tarfile_to_samples(),
                        wds.shuffle(500,initial=500),
                        wds.decode("torch"),
                        wds.rename(images="jpg;png", voxels="nsdgeneral.npy", 
                                    embs="sgxl_emb.npy", trial="trial.npy"),
                        wds.to_tuple("voxels", image_var),
                        wds.batched(batch_size, partial=True),
                    ]).with_epoch(num_worker_batches)
    train_dl = wds.WebLoader(train_data, num_workers=num_workers,
                            batch_size=None, shuffle=False, persistent_workers=True)

    # Validation #
    num_samples = 492
    num_batches = math.ceil(num_samples / global_batch_size)
    num_worker_batches = math.ceil(num_batches / num_workers)
    print("validation: num_worker_batches",num_worker_batches)

    val_data = wds.DataPipeline([wds.SimpleShardList(val_url),
                        wds.tarfile_to_samples(),
                        wds.decode("torch"),
                        wds.rename(images="jpg;png", voxels="nsdgeneral.npy", 
                                    embs="sgxl_emb.npy", trial="trial.npy"),
                        wds.to_tuple("voxels", image_var),
                        wds.batched(batch_size, partial=True),
                    ]).with_epoch(num_worker_batches)
    val_dl = wds.WebLoader(val_data, num_workers=num_workers,
                        batch_size=None, shuffle=False, persistent_workers=True)

    return train_dl, val_dl

def load_sd_pipeline():

    from diffusers import StableDiffusionImageVariationPipeline
    from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
    # from transformers import CLIPVisionModelWithProjection, CLIPFeatureExtractor
    
    cache_dir = "lambdalabs/sd-image-variations-diffusers"
    
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        cache_dir, 
        revision="v2.0"
    ).to(device)

    assert sd_pipe.image_encoder.training == False, 'not in eval mode'

    # unet = UNet2DConditionModel.from_pretrained(cache_dir, subfolder="unet").to(device)
    # vae = AutoencoderKL.from_pretrained(cache_dir, subfolder="vae").to(device)
    # noise_scheduler = PNDMScheduler.from_pretrained(cache_dir, subfolder="scheduler")
    unet = sd_pipe.unet
    vae = sd_pipe.vae
    noise_scheduler = sd_pipe.scheduler

    unet.eval() # dont want to train model
    unet.requires_grad_(False) # dont need to calculate gradients

    vae.eval() # dont want to train model
    vae.requires_grad_(False) # dont need to calculate gradients

    return sd_pipe, unet, vae, noise_scheduler

@torch.no_grad()
def denoising_loop(
    unet, noise_scheduler, num_inference_steps, num_per_sample, encoder_hidden_states,
    guidance_scale, latents=None,
):
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    if latents is None:
        print("generating latents")
        latents = torch.randn([num_per_sample, 4, 64, 64], device=device)
    
    # Denoising loop (original clip)
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    return latents

#@torch.no_grad()
def sample_images(
    clip_extractor, brain_net, unet, vae, noise_scheduler, diffusion_prior, voxel, img_input,
    num_inference_steps=50, clip_guidance_scale=7.5, vox_guidance_scale=7.5, num_per_sample=4,
    prior_timesteps=None,
    seed=None,
):
    print('seed', seed)
    
    assert voxel.shape[0] == img_input.shape[0], 'batches must be the same for voxels and images'
    n_examples = voxel.shape[0]

    clip_extractor.eval()
    brain_net.eval()

    unet.eval()
    unet.requires_grad_(False)

    vae.eval()
    vae.requires_grad_(False)

    diffusion_prior.eval()
    # diffusion_prior.requires_grad_(False)

    def decode_latents(latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image[0]

    # tform = transforms.Compose([
    #     #transforms.ToTensor(), ## don't need this since we've already got tensors
    #     transforms.Resize(
    #         (224, 224),
    #         interpolation=transforms.InterpolationMode.BICUBIC,
    #         antialias=False,
    #         ),
    #     transforms.Normalize(
    #     [0.48145466, 0.4578275, 0.40821073],
    #     [0.26862954, 0.26130258, 0.27577711]),
    # ])

    for idx in range(n_examples):
        if seed is not None:
            seed_everything(seed)

        image = clip_extractor.resize_image(img_input[[idx]])
        
        # Original clip embedding:
        clip_emb = clip_extractor.embed_image(image)
        # clip_emb = sd_pipe._encode_image(tform(image), device, 1, False).squeeze(1)
        # print('clip_emb', clip_emb.shape)
        norm_orig = clip_emb.norm().item()

        # Encode voxels to CLIP space
        image_embeddings = brain_net(voxel[[idx]].to(device).float())
        norm_pre_prior = image_embeddings.norm().item()
        
        # image_embeddings = nn.functional.normalize(image_embeddings, dim=-1) 
        # image_embeddings *= clip_emb[1].norm()/image_embeddings.norm() # note: this is cheating to equate norm scaling
        image_embeddings = diffusion_prior.p_sample_loop(image_embeddings.shape, 
                                            text_cond = dict(text_embed = image_embeddings), 
                                            cond_scale = 1., timesteps = prior_timesteps)
        norm_post_prior = image_embeddings.norm().item()
        cos_sim = nn.functional.cosine_similarity(image_embeddings, clip_emb, dim=1).item()
        
        print(f"Cosine sim: {cos_sim:.3f}")
        print(f"norms | orig: {norm_orig:.3f}, pre_prior: {norm_pre_prior:.3f}, post_prior: {norm_post_prior:.3f}")

        plt.plot(clip_emb.detach().cpu().numpy().flatten(),label='CLIP-image emb.')
        plt.plot(image_embeddings.detach().cpu().numpy().flatten(),label='CLIP-voxel emb.')
        plt.title('MSE: %.5f' % nn.functional.mse_loss(image_embeddings, clip_emb).item())
        plt.legend()
        plt.show()

        # duplicate the embedding to serve classifier free guidance
        image_embeddings = image_embeddings.repeat(num_per_sample, 1)
        image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings]).unsqueeze(1).to(device)

        # Prepare timesteps
        # noise_scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = noise_scheduler.timesteps
        latents = torch.randn([num_per_sample, 4, 64, 64], device=device)

        # duplicate the embedding to serve classifier free guidance
        clip_emb = torch.cat([torch.zeros_like(clip_emb), clip_emb]).unsqueeze(1).to(device).float()
        # clip_latents = torch.randn([1, 4, 64, 64], device=device)
        clip_latents = latents[0].unsqueeze(0).clone() # use the same latent as the first brain image

        if seed is not None:
            seed_everything(seed)

        clip_latents = denoising_loop(
            unet, noise_scheduler, num_inference_steps, 1, clip_emb, clip_guidance_scale, 
            latents=clip_latents
        )

        if seed is not None:
            seed_everything(seed)

        latents = denoising_loop(
            unet, noise_scheduler, num_inference_steps, num_per_sample, image_embeddings, vox_guidance_scale, 
            latents=latents
        )
        
        # # Denoising loop (original clip)
        # for i, t in enumerate(timesteps):
        #     # expand the latents if we are doing classifier free guidance
        #     latent_model_input = torch.cat([clip_latents] * 2)
        #     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        #     # predict the noise residual
        #     noise_pred = unet(latent_model_input, t, encoder_hidden_states=clip_emb).sample
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + clip_guidance_scale * (noise_pred_text - noise_pred_uncond)

        #     # compute the previous noisy sample x_t -> x_t-1
        #     clip_latents = noise_scheduler.step(noise_pred, t, clip_latents).prev_sample
        
        # decoded_clip_image = decode_latents(clip_latents)
        
        # # Denoising loop (brain recon)
        # for i, t in enumerate(timesteps):
        #     # expand the latents if we are doing classifier free guidance
        #     latent_model_input = torch.cat([latents] * 2)
        #     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        #     # predict the noise residual
        #     noise_pred = unet(latent_model_input, t, encoder_hidden_states=image_embeddings).sample
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + vox_guidance_scale * (noise_pred_text - noise_pred_uncond)

        #     # compute the previous noisy sample x_t -> x_t-1
        #     latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        fig, ax = plt.subplots(1, 2+num_per_sample, figsize=(23,3))
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image))
        ax[1].set_title(f"Recon from original CLIP")
        ax[1].imshow(torch_to_Image(decode_latents(clip_latents)))
        for i in range(2, 2+num_per_sample):
            recon = decode_latents(latents[i-2].unsqueeze(0))
            ax[i].set_title(f"Recon {i-1} from brain")
            ax[i].imshow(torch_to_Image(recon))
        for i in range(2+num_per_sample):
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()       
        