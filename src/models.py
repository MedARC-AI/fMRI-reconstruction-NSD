import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
import clip

# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from tqdm.auto import tqdm
import random
import json
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig

# for pipeline
from diffusers import StableDiffusionImageVariationPipeline
from typing import Callable, List, Optional, Union

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False, train_transforms=None, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32"
        print(clip_variant, device)
        clip_model, _ = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clip_size = (224,224)
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.transforms = train_transforms
        self.device= device

    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return nn.functional.interpolate(image.to(self.device), self.clip_size, mode="area", antialias=False)

    def embed_image(self, image, return_norm=False):
        """Expects images in -1 to 1 range"""
        clip_emb = self.resize_image(image)
        if self.transforms is not None:
            clip_emb = self.transforms(clip_emb)
        clip_emb = self.clip.encode_image(self.normalize(clip_emb))
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        clip_emb_norm = clip_emb.norm(2,dim=-1)
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        if not return_norm:
            return clip_emb
        else:
             return clip_emb, clip_emb_norm
    
    def embed_text(self, text_samples):
        clip_text = clip.tokenize(text_samples).to(self.device)
        clip_text = self.clip.encode_text(clip_text)
        if self.clamp_embs:
            clip_text = torch.clamp(clip_text, -1.5, 1.5)
        if self.norm_embs:
            clip_text = nn.functional.normalize(clip_text, dim=-1)
        return clip_text
    
    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)

class BrainNetwork(nn.Module):
    # 133M
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=4):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(h),
            nn.Dropout(0.5),
        )
            
        
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        
        self.lin1 = nn.Linear(h, out_dim)
        self.n_blocks = n_blocks
        
    def forward(self, x):
        '''
            bs, 1, 15724 -> bs, 32, h
            bs, 32, h -> bs, 32h
            b2, 32h -> bs, 768
        '''
        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        return x
    
class BrainNetworkLarge(nn.Module):
    # 235M
    def __init__(self, out_dim, in_dim=15724, h=4096):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.15) if i!=0 else nn.Identity(),
                nn.Linear(h, h),
                nn.GELU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.25),
                # nn.Dropout(0.15),
                nn.Linear(h, h),
                nn.GELU(),
                nn.BatchNorm1d(h),
                # nn.Dropout(0.15),
            ) for i in range(5)
        ])  
        
        # zero init batchnorms
        for lin in self.lins:
            nn.init.constant_(lin[-1].weight, 0.0)
            # nn.init.constant_(lin[-2].weight, 0.0)
        
        self.lin1 = nn.Linear(4096, out_dim)
        
    def forward(self, x):
        x = self.conv(x)  # bs, 4096
        residual = x
        for lin in self.lins:
            x = lin(x)
            x += residual
            residual = x
        x = self.lin1(x)
        return x

class BrainDiffusionPrior(DiffusionPrior):
    """ 
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.,
                generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            #noise = torch.randn_like(x)
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, 
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss, pred

    def forward(
        self,
        text = None,
        image = None,
        voxel = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)

        return loss, pred, text_embed
   
    @staticmethod
    def from_pretrained(net_kwargs={}, prior_kwargs={}):
        # "https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json"
        config_url = "checkpoints/prior_config.json"
        config = json.load(open(config_url))
        
        config['prior']['net']['max_text_len'] = 256
        config['prior']['net'].update(net_kwargs)
        print('net_config', config['prior']['net'])
        net_config = DiffusionPriorNetworkConfig(**config['prior']['net'])

        kwargs = config['prior']
        kwargs.pop('clip')
        kwargs.pop('net')
        kwargs.update(prior_kwargs)
        print('prior_config', kwargs)

        diffusion_prior_network = net_config.create()
        diffusion_prior = BrainDiffusionPrior(net=diffusion_prior_network, clip=None, **kwargs).to(torch.device('cpu'))
        
        # 'https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth'
        ckpt_url = 'checkpoints/best.pth'
        ckpt = torch.load(ckpt_url, map_location=torch.device('cpu'))

        # Note these keys will be missing (maybe due to an update to the code since training):
        # "net.null_text_encodings", "net.null_text_embeds", "net.null_image_embed"
        # I don't think these get used if `cond_drop_prob = 0` though (which is the default here)
        keys = diffusion_prior.load_state_dict(ckpt, strict=False)
        print("missing keys in prior checkpoint (probably ok)", keys.missing_keys)
        
        return diffusion_prior

class BrainSD(StableDiffusionImageVariationPipeline):
    """ 
    Differences from original:
    - Keep generated images on GPU and return tensors
    - No NSFW checker
    - Can pass in image or image_embedding to generate a variation
    NOTE: requires latest version of diffusers to avoid the latent dims not being correct.
    """

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        image_embeddings: Optional[torch.FloatTensor] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if image_embeddings is None:
            assert image is not None, "If image_embeddings is None, image must not be None"
    
            # resize and normalize the way that's recommended
            tform = transforms.Compose([
                #transforms.ToTensor(), ## don't need this since we've already got tensors
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=False,
                    ),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
            ])
            image = tform(image)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(image, height, width, callback_steps)

            # 2. Define call parameters
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            else:
                batch_size = image.shape[0]

            # 3. Encode input image
            image_embeddings = self._encode_image(image, device, num_images_per_prompt, do_classifier_free_guidance)
        else:
            batch_size = image_embeddings.shape[0] // 2

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=image_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, image_embeddings.dtype)

        # # 10. Convert to PIL
        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image)

        return image
