from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
import json
from typing import Tuple, List, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, dropout: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        layers = [("c_fc", nn.Linear(d_model, mlp_width))]
        if dropout > 0.0:
            layers.append(("c_drop", nn.Dropout(dropout)))
        layers.extend([("c_act", act_layer()), ("proj", nn.Linear(mlp_width, d_model))])
        self.mlp = nn.Sequential(OrderedDict(layers))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, dropout: float = 0.0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, dropout=dropout)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
    
class NewVoxel3dConvEncoder(nn.Module):
    def __init__(self, dims: List[int], attention_width: int, output_dim: int, c_in: int = 1, average_output: bool = False, act_layer: Callable = nn.GELU):
        super().__init__()
        self.average_output = average_output  # Average the output of the transformer instead of using a flattened linear layer
        self.channels = [64, 128, 256, 256, 256, attention_width]
        self.strides = [1, 1, 1, 2, 2, 2]
        self.padding = [1, 1, 1, 0, 1, 0]
        self.dialation = [1, 1, 1, 1, 1, 1]
        self.kernel = [3, 3, 3, 3, 3, 3]
        assert len(self.channels) == len(self.strides) == len(self.padding) == len(self.dialation) == len(self.kernel), f"Lengths of channels, strides, padding, dialation, and kernel must be the same. Got {len(self.channels)}, {len(self.strides)}, {len(self.padding)}, {len(self.dialation)}, {len(self.kernel)}"
        channels = [c_in] + self.channels
        self.conv_blocks = nn.ModuleList([
            self._get_conv_layer(channels[i], channels[i + 1], kernel_size=self.kernel[i], stride=self.strides[i], padding=self.padding[i], dilation=self.dialation[i], act_layer=act_layer)
            for i in range(len(self.channels))
        ])
        for n in range(len(self.channels)):
            stride = self.strides[n]
            dialation = self.dialation[n]
            padding = self.padding[n]
            kernel = self.kernel[n]
            dims = [int((d + 2*padding - dialation*(kernel - 1) - 1)/(stride) + 1) for d in dims]
            logging.info(f"Conv {n} output shape: {dims}")
        logging.info(f"Transformer sequence length: {np.prod(dims)}. Transformer width: {attention_width}")
        self.transformer = Transformer(attention_width, layers=2, heads=8, mlp_ratio=4, act_layer=act_layer, dropout=0.0)
        logging.info(f"Projection input features: {attention_width * dims[0] * dims[1] * dims[2]}")
        if self.average_output:
            self.proj = nn.Parameter(attention_width**-0.5 * torch.randn(attention_width, output_dim))
        else:
            # self.proj = nn.Linear(attention_width * dims[0] * dims[1] * dims[2], output_dim)
            self.proj = nn.LazyLinear(output_dim)

    def _get_conv_layer(self, c_in, c_out, kernel_size, stride, padding, dilation, act_layer):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.Dropout3d(0.1),
            act_layer(),
            # nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor):
        for block in self.conv_blocks:
            x = block(x)
        # Currently the output shape is [*, attention_width, x, y, z]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [*, attention_width, seq_len]
        x = x.permute(2, 0, 1) # [seq_len, *, attention_width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # [*, seq_len, attention_width]
        if self.average_output:
            x = x.mean(dim=1)
            x = x @ self.proj
        else:
            x = x.reshape(x.shape[0], -1) # [*, attention_width * seq_len]
            x = self.proj(x)
        return x
    
if __name__ == "__main__":
    # quick_gelu = True
    # act_layer = QuickGELU if quick_gelu else nn.GELU
    # voxel_config = json.load(open('ViT-L14_3d.json', 'r'))
    model = NewVoxel3dConvEncoder(
                dims=[42, 46, 61], # voxel_config['config_3d_conv']["dims"],
                attention_width=64,
                output_dim=768,
                average_output=False,
                # act_layer=act_layer
            )