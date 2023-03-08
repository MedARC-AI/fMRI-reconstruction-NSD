"""
Conv3D model extracted from @aidan's OpenCLIP fork by @Atom_101.
"""
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
    def __init__(self, d_model: int, n_head: int, 
        mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, dropout: float = 0.0
    ):
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
    def __init__(self, width: int, layers: int, heads: int,
        mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, dropout: float = 0.0
    ):
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
                x = torch.utils.checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

class NewVoxel3dConvEncoder(nn.Module):
    def __init__(self, dims: List[int] = [81, 104, 83], attention_width: int = 64, out_dim: int = 768, 
        c_in: int = 1, average_output: bool = False, act_layer: Callable = nn.GELU,
        channels: Optional[List[int]] = None, strides: Optional[List[int]] = None,
        padding: Optional[List[int]] = None, dilation: Optional[List[int]] = None,
        kernel: Optional[List[int]] = None,
        input_dropout_prob=0.1,
    ):
        super().__init__()
        
        # Average the output of the transformer instead of using a flattened linear layer
        self.average_output = average_output
        
        if channels is None:
            channels = [64, 128, 256, 256, 256, attention_width]
        if strides is None:
            strides = [1, 1, 1, 2, 2, 2]
        if padding is None:
            padding = [1, 1, 1, 0, 1, 0]
        if dilation is None:
            dilation = [1, 1, 1, 1, 1, 1]
        if kernel is None:
            kernel = [3, 3, 3, 3, 3, 3]

        self.channels = channels
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.kernel = kernel

        assert len(self.channels) == len(self.strides) == len(self.padding) == len(self.dilation) == len(self.kernel), \
            f"Lengths of channels, strides, padding, dilation, and kernel must be the same. " \
            f"Got {len(self.channels)}, {len(self.strides)}, {len(self.padding)}, {len(self.dilation)}, {len(self.kernel)}"

        channels = [c_in] + self.channels

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels[i], channels[i + 1], self.kernel[i], 
                          stride=self.strides[i], padding=self.padding[i], dilation=self.dilation[i]),
                nn.BatchNorm3d(channels[i + 1]),
                act_layer(),
                nn.Dropout3d(p=0.2),
                # nn.MaxPool3d(kernel_size=2, stride=2)
            )
            for i in range(len(self.channels))
        ])

        print(f"Input shape: {dims}")
        for n in range(len(self.channels)):
            stride = self.strides[n]
            dilation = self.dilation[n]
            padding = self.padding[n]
            kernel = self.kernel[n]
            dims = [int((d + 2*padding - dilation*(kernel - 1) - 1)/(stride) + 1) for d in dims]
            print(f"Conv {n} output: {channels[n + 1]} x {dims}")
        
        print(f"Transformer sequence length: {np.prod(dims)}. Transformer width: {attention_width}")
        
        self.transformer = Transformer(attention_width, layers=2, heads=8, mlp_ratio=4, 
            act_layer=act_layer, dropout=0.0
        )
        
        print(f"Projection input features: {attention_width * dims[0] * dims[1] * dims[2]}")
        
        if self.average_output:
            self.proj = nn.Parameter(attention_width**-0.5 * torch.randn(attention_width, out_dim))
        else:
            # 69120
            self.proj = nn.Linear(attention_width * dims[0] * dims[1] * dims[2], out_dim)
            # cant use lazy with ddp :/
            # self.proj = nn.LazyLinear(out_dim)

        self.input_dropout = nn.Dropout3d(p=input_dropout_prob)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, f"Input must be 4D. Got {x.ndim}D"
        
        # add singleton channel dimension
        x = x.unsqueeze(1)

        # dropout on raw voxel inputs
        x = self.input_dropout(x)

        for block in self.conv_blocks:
            x = block(x)

        import pdb; pdb.set_trace()

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

class SimpleVoxel3dConvEncoder(nn.Module):
    def __init__(self, dims: List[int] = [81, 104, 83], out_dim: int = 768, 
        c_in: int = 1, act_layer: Callable = nn.GELU,
        channels: Optional[List[int]] = None, strides: Optional[List[int]] = None,
        padding: Optional[List[int]] = None, dilation: Optional[List[int]] = None,
        kernel: Optional[List[int]] = None,
        input_dropout_prob=0.1,
    ):
        super().__init__()
        
        self.channels = channels
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.kernel = kernel

        assert len(self.channels) == len(self.strides) == len(self.padding) == len(self.dilation) == len(self.kernel), \
            f"Lengths of channels, strides, padding, dilation, and kernel must be the same. " \
            f"Got {len(self.channels)}, {len(self.strides)}, {len(self.padding)}, {len(self.dilation)}, {len(self.kernel)}"

        channels = [c_in] + self.channels

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels[i], channels[i + 1], self.kernel[i], 
                          stride=self.strides[i], padding=self.padding[i], dilation=self.dilation[i]),
                nn.BatchNorm3d(channels[i + 1]),
                act_layer(),
                nn.Dropout3d(p=0.2),
                # nn.MaxPool3d(kernel_size=3, stride=2)
            )
            for i in range(len(self.channels))
        ])

        print(f"Input shape: {dims}")
        for n in range(len(self.channels)):
            stride = self.strides[n]
            dilation = self.dilation[n]
            padding = self.padding[n]
            kernel = self.kernel[n]
            dims = [int((d + 2*padding - dilation*(kernel - 1) - 1)/(stride) + 1) for d in dims]
            print(f"Conv {n} output: {channels[n + 1]} x {dims}", flush=True)
        
        n_feats = channels[-1] * np.prod(dims)
        print(f"Projection input features: {n_feats}", flush=True)
        
        self.proj = nn.Linear(n_feats, out_dim)
        self.input_dropout = nn.Dropout3d(p=input_dropout_prob)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, f"Input must be 4D. Got {x.ndim}D"
        
        # add singleton channel dimension
        x = x.unsqueeze(1)

        # dropout on raw voxel inputs
        x = self.input_dropout(x)

        for block in self.conv_blocks:
            # print(x.shape)
            x = block(x)
            # print('\t', x.shape)

        # import pdb; pdb.set_trace()

        # [*, c_out_last, x, y, z] -> [*, ]
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x
    
if __name__ == "__main__":
    # quick_gelu = True
    # act_layer = QuickGELU if quick_gelu else nn.GELU
    # voxel_config = json.load(open('ViT-L14_3d.json', 'r'))
    model = NewVoxel3dConvEncoder(
                dims=[42, 46, 61], # voxel_config['config_3d_conv']["dims"],
                attention_width=64,
                out_dim=768,
                average_output=False,
                # act_layer=act_layer
            )