import os
from functools import partial
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import clip
import open_clip

# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from tqdm.auto import tqdm
import random
import json
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig

# FOr VD prior
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward

# for pipeline
from diffusers import StableDiffusionImageVariationPipeline, VersatileDiffusionDualGuidedPipeline
from typing import Callable, List, Optional, Union

from diffusers.models.vae import Decoder

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class OpenClipper(torch.nn.Module):
#     def __init__(self, clip_variant='ViT-H-14', train_transforms=None, device=torch.device('cpu')):
#         super().__init__()
#         print(clip_variant, device)
#         assert clip_variant == 'ViT-H-14' # not setup for other models yet
                
#         clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', 
#                                         pretrained='laion2b_s32b_b79k', device=device)
#         clip_model.eval() # dont want to train model
#         for param in clip_model.parameters():
#             param.requires_grad = False # dont need to calculate gradients
            
#         # overwrite preprocess to accept torch inputs instead of PIL Image
#         preprocess = transforms.Compose([
#                 transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
#                 transforms.CenterCrop(224),
#                 transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
#         ])
        
#         tokenizer = open_clip.get_tokenizer('ViT-H-14')
            
#         self.clip = clip_model
#         self.preprocess = preprocess
#         self.tokenizer = tokenizer
#         self.transforms = train_transforms
#         self.device = device
        
#     def embed_image(self, image):
#         """Expects images in -1 to 1 range"""
#         image = self.preprocess(image).to(self.device)
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             image_features = self.clip.encode_image(image)
#             #image_features /= image_features.norm(dim=-1, keepdim=True)
#         return image_features

#     def embed_text(self, text_samples):
#         text = self.tokenizer(text_samples).to(self.device)
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             text_features = self.clip.encode_text(text)
#             #text_features /= text_features.norm(dim=-1, keepdim=True)
#         return text_features

#     def embed_curated_annotations(self, annots):
#         for i,b in enumerate(annots):
#             t = ''
#             while t == '':
#                 rand = torch.randint(5,(1,1))[0][0]
#                 t = b[0,rand]
#             if i==0:
#                 txt = np.array(t)
#             else:
#                 txt = np.vstack((txt,t))
#         txt = txt.flatten()
#         return self.embed_text(txt)

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False, refine=False, train_transforms=None, 
                 hidden_state=False, token_idx=None, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            # from transformers import CLIPVisionModelWithProjection
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir="/fsx/proj-medarc/fmri/cache")
            from transformers import CLIPVisionModelWithProjection
            sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').to(device)
            image_encoder.eval()
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.image_encoder = image_encoder
            self.refine = refine
            self.token_idx = token_idx
        else:
            clip_model, preprocess = clip.load(clip_variant, device=device)
            clip_model.eval() # dont want to train model
            for param in clip_model.parameters():
                param.requires_grad = False # dont need to calculate gradients        
            self.clip = clip_model
        
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())

        self.hidden_state = hidden_state
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.transforms = train_transforms
        self.device= device
        
    def versatile_process_embeddings(self, encoder_output):
        embeds = encoder_output.last_hidden_state
        if not self.refine:
            embeds = self.image_encoder.vision_model.post_layernorm(embeds)
            embeds = self.image_encoder.visual_projection(embeds)
            if self.norm_embs:
                # normalize all tokens by cls token's norm
                norm = torch.norm(embeds[:, 0], dim=-1).reshape(-1, 1, 1) + 1e-6
                embeds = embeds/norm
        if self.token_idx is not None:
            embeds = embeds[:,self.token_idx]
        return embeds

    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return nn.functional.interpolate(image.to(self.device), self.clip_size, mode="bilinear", 
            align_corners=False, antialias=True
        )

    @torch.no_grad()
    def embed_image(self, image, apply_transforms=True, apply_spatial_transforms=True):
        """Expects images in -1 to 1 range"""
        if self.transforms is not None and apply_transforms:
            if isinstance(self.transforms, list):
                # clip_emb = self.transforms[0](image.cpu())
                clip_emb = self.transforms[0](image)
                if apply_spatial_transforms:
                    # clip_emb = self.transforms[1](clip_emb.cpu())
                    clip_emb = self.transforms[1](clip_emb)
            else:
                clip_emb = self.transforms(image)
        else:
            clip_emb = image
        clip_emb = self.resize_image(clip_emb.to(self.device))
        clip_emb = self.normalize(clip_emb)
        
        if self.clip_variant=="ViT-L/14" and self.hidden_state:
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_process_embeddings(clip_emb)
            return clip_emb
        
        clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb

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

class OpenClipper(Clipper):
    def __init__(self, clip_variant, weights_path, clamp_embs=False, norm_embs=False, train_transforms=None, device=torch.device('cpu')):
        torch.nn.Module.__init__(self)
        print(clip_variant, device)
        clip_model, _, _ = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained=False, device=torch.device('cuda'))
        clip_model.load_state_dict(torch.load(weights_path))
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clip_size = (256,256) if 'convnext' in clip_variant else (224, 224)
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.transforms = train_transforms
        self.device= device

class BrainNetwork(nn.Module):
    # 133M
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=4, norm_type='bn', act_first=True):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        self.temp = nn.Parameter(torch.tensor(.006))
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
            
        
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        
    def forward(self, x):
        '''
            bs, 1, 15724 -> bs, 32, h
            bs, 32, h -> bs, 32h
            b2, 32h -> bs, 768
        '''
        if x.ndim == 4:
            # case when we passed 3D data of shape [N, 81, 104, 83]
            assert x.shape[1] == 81 and x.shape[2] == 104 and x.shape[3] == 83
            # [N, 699192]
            x = x.reshape(x.shape[0], -1)

        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        return x

class BrainNetworkDETR(BrainNetwork):
    # 250M
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=4, norm_type='bn', act_first=True, 
                encoder_tokens=32, decoder_tokens=257):
        # encoder
        super().__init__(out_dim*encoder_tokens, in_dim, h, n_blocks, norm_type, act_first)
        self.norm = nn.LayerNorm(out_dim)
        self.encoder_tokens = encoder_tokens
        
        self.register_parameter('queries', nn.Parameter(torch.randn(1, decoder_tokens, out_dim)))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=out_dim, nhead=8,
                                        dim_feedforward=1024, 
                                        batch_first=True, dropout=0.25),
            num_layers=n_blocks
        )
        self.decoder_projector = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim)
        )


    def forward(self, x):
        enc = super().forward(x)
        enc = self.norm(enc.reshape(enc.shape[0], self.encoder_tokens, -1))

        dec = self.transformer(self.queries.expand(x.shape[0], -1, -1), enc)
        dec = self.decoder_projector(dec)
        return dec

class BrainNetworkDETR2(BrainNetwork):
    # 182M
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=4, norm_type='bn', act_first=True, 
                encoder_tokens=8, decoder_tokens=257, use_projector=False):
        # encoder
        super().__init__(out_dim*encoder_tokens, in_dim, h, n_blocks, norm_type, act_first)
        self.norm = nn.LayerNorm(out_dim)
        self.encoder_tokens = encoder_tokens
        
        self.register_parameter('queries', nn.Parameter(torch.randn(1, decoder_tokens, out_dim)))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=out_dim, nhead=8,
                                       dim_feedforward=1024, norm_first=True,
                                       activation='gelu',batch_first=True, 
                                       dropout=0.25),
            num_layers=n_blocks
        )
        self.decoder_projector = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, out_dim)
            )


    def forward(self, x):
        enc = super().forward(x)
        enc = self.norm(enc.reshape(enc.shape[0], self.encoder_tokens, -1))

        dec = self.transformer(self.queries.expand(x.shape[0], -1, -1), enc)
        dec = self.decoder_projector(dec)
        if self.use_projector:
            return dec, self.projector(dec)
        return dec

class BrainNetworkNoDETR(BrainNetwork):
    # 950M
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=4, norm_type='bn', act_first=True, 
                encoder_tokens=257, decoder_tokens=257, use_projector=False):
        # encoder
        super().__init__(out_dim*encoder_tokens, in_dim, h, n_blocks, norm_type, act_first)
        # self.norm = nn.LayerNorm(out_dim)
        self.encoder_tokens = encoder_tokens
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, out_dim)
            )

    def forward(self, x):
        enc = super().forward(x)
        enc = enc.reshape(enc.shape[0], self.encoder_tokens, -1)
        if self.encoder_tokens == 1:
            enc = enc.squeeze(1)
        if self.use_projector:
            return enc, self.projector(enc)
        return enc

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

class VersatileDiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None,
        num_time_embeds = 1,
        # num_image_embeds = 1,
        # num_brain_embeds = 1,
        num_tokens = 257,
        causal = True,
        learned_query_mode = 'none',
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens*2+1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob = 0.,
        text_cond_drop_prob = None,
        image_cond_drop_prob = 0.
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        
        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds
        
        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device = device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b = batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        
        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim = -2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed

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

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
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
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

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

        # return denormalized pred, diff model learns to predict normalized pred
        return loss, pred/self.image_embed_scale, (clip_voxels_mse, clip_voxels)
   
    @staticmethod
    def from_pretrained(net_kwargs={}, prior_kwargs={}, voxel2clip_path=None, ckpt_dir='./checkpoints'):
        # "https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json"
        config_url = os.path.join(ckpt_dir, "prior_config.json")
        config = json.load(open(config_url))
        
        config['prior']['net']['max_text_len'] = 256
        config['prior']['net'].update(net_kwargs)
        # print('net_config', config['prior']['net'])
        net_config = DiffusionPriorNetworkConfig(**config['prior']['net'])

        kwargs = config['prior']
        kwargs.pop('clip')
        kwargs.pop('net')
        kwargs.update(prior_kwargs)
        # print('prior_config', kwargs)

        diffusion_prior_network = net_config.create()
        diffusion_prior = BrainDiffusionPrior(net=diffusion_prior_network, clip=None, **kwargs).to(torch.device('cpu'))
        
        # 'https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth'
        ckpt_url = os.path.join(ckpt_dir, 'best.pth')
        ckpt = torch.load(ckpt_url, map_location=torch.device('cpu'))

        # Note these keys will be missing (maybe due to an update to the code since training):
        # "net.null_text_encodings", "net.null_text_embeds", "net.null_image_embed"
        # I don't think these get used if `cond_drop_prob = 0` though (which is the default here)
        diffusion_prior.load_state_dict(ckpt, strict=False)
        # keys = diffusion_prior.load_state_dict(ckpt, strict=False)
        # print("missing keys in prior checkpoint (probably ok)", keys.missing_keys)

        if voxel2clip_path:
            # load the voxel2clip weights
            checkpoint = torch.load(voxel2clip_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['model_state_dict']
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict[key]
                    del state_dict[key]
            diffusion_prior.voxel2clip.load_state_dict(state_dict)
        
        return diffusion_prior

class BrainVD(VersatileDiffusionDualGuidedPipeline):
    """ 
    Differences from original:
    - Keep generated images on GPU and return tensors
    - No NSFW checker
    - Can pass in image or image_embedding to generate a variation
    NOTE: requires latest version of diffusers to avoid the latent dims not being correct.
    """

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(self, prompt, image, height, width, callback_steps):
        if prompt is not None and not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type None, `str` or `list` but is {type(prompt)}")
        if image is not None and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(f"`image` has to be of type None, `PIL.Image` or `list` but is {type(image)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        image: Union[str, List[str]] = None,
        text_to_image_strength: float = 0.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        image_embeddings: Optional[torch.FloatTensor] = None,
        prompt_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):

        height = height or self.image_unet.config.sample_size * self.vae_scale_factor
        width = width or self.image_unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, image, height, width, callback_steps)

        prompt = [prompt] if prompt is not None and not isinstance(prompt, list) else prompt
        image = [image] if image is not None and not isinstance(image, list) else image
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # 3. Encode input prompt
        if image_embeddings is None:
            if image is not None:
                image_embeddings = self._encode_image_prompt(
                    image, device, num_images_per_prompt, do_classifier_free_guidance
                )
                batch_size = len(image)
            else:
                image_embeddings = None
        
        if prompt_embeddings is None:
            if prompt is not None:
                prompt_embeddings = self._encode_text_prompt(
                    prompt, device, num_images_per_prompt, do_classifier_free_guidance
                )
                batch_size = len(prompt)
            else:
                prompt_embeddings = None
        if image_embeddings is not None:
            batch_size = image_embeddings.shape[0] // 2
        elif prompt_embeddings is not None:
            batch_size = prompt_embeddings.shape[0] // 2
        
        if image_embeddings is not None and prompt_embeddings is not None:
            dual_prompt_embeddings = torch.cat([prompt_embeddings, image_embeddings], dim=1)
        elif image_embeddings is None:
            dual_prompt_embeddings = prompt_embeddings
            text_to_image_strength = 1.
        elif prompt_embeddings is None:
            dual_prompt_embeddings = image_embeddings
            text_to_image_strength = 0.
        else:
            raise ValueError()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.image_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dual_prompt_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Combine the attention blocks of the image and text UNets
        self.set_transformer_params(text_to_image_strength, ("text", "image"))

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.image_unet(latent_model_input, t, encoder_hidden_states=dual_prompt_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        return image

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

class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=False):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, 16384, bias=False)
        self.norm = nn.LayerNorm(512)

        self.register_parameter('queries', nn.Parameter(torch.randn(1, 256, 512)))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, 
                                        dim_feedforward=1024, 
                                        batch_first=True, dropout=0.25),
            num_layers=n_blocks
        )

        # option 1  -> 124.56M
        # self.lin1 = nn.Linear(h, 32768, bias=True)
        # self.upsampler = Decoder(
        #     in_channels=64,
        #     out_channels=4,
        #     up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
        #     block_out_channels=[64, 128, 256, 256],
        #     layers_per_block=1,
        # )

        # option2  -> 132.52M
        # self.lin1 = nn.Linear(h, 1024, bias=True)
        # self.upsampler = Decoder(
        #     in_channels=64,
        #     out_channels=4,
        #     up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
        #     block_out_channels=[64, 128, 256, 256, 512],
        #     layers_per_block=1,
        # )
        
        if use_cont:
            self.maps_projector = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(True),
                nn.Linear(512, 512)
            )
        else:
            self.maps_projector = nn.Identity()

        self.upsampler = nn.Sequential(
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 320, 3, padding=1),
            nn.GroupNorm(32, 320),
            nn.SiLU(inplace=True),
            nn.Conv2d(320, 320, 3, padding=1),
            nn.GroupNorm(32, 320),
            nn.SiLU(inplace=True),
            nn.Conv2d(320, 4, 3, padding=1)
        )

    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        # # x = x.reshape(x.shape[0], -1, 8, 8).contiguous()  # bs, 64, 8, 8
        # x = x.reshape(x.shape[0], -1, 64, 64).contiguous()
        # return self.upsampler(x)

        # decoder
        x = self.norm(x.reshape(x.shape[0], 32, 512))
        preds = self.transformer(self.queries.expand(x.shape[0], -1, -1), x)
        sd_embeds = preds.permute(0,2,1).reshape(-1, 512, 16, 16)
        sd_embeds = F.pixel_shuffle(sd_embeds, 4)  # bs, 32, 32, 32

        # contrastive
        if return_transformer_feats:
            return self.upsampler(sd_embeds), self.maps_projector(preds)
        
        return self.upsampler(sd_embeds)
