import io
import math
import re
from functools import partial
from typing import List

import albumentations as A
import cv2
import numpy as np
import pyarrow as pa
import requests
import torch
import transformers
from PIL import Image
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoProcessor
from transformers.activations import ACT2FN

assert transformers.__version__ == "4.40.0", "Please install a specific HF transformers version: pip install transformers==4.40.0"


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        # trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask
        )[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class CLIPModel(nn.Module):

    def __init__(
            self,
            image_size: int,
            n_queries: int = 256,
            output_dim: int = 512,
            vision_model_name_or_path: str = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli",
            **kwargs
    ):
        super().__init__()
        # load model and processor
        self.model = AutoModel.from_pretrained(vision_model_name_or_path).vision_model
        self.processor = AutoProcessor.from_pretrained(vision_model_name_or_path).image_processor

        # set constants
        self.image_height, self.image_width = self.image_size = (image_size, image_size)
        width = self.model.config.hidden_size
        patch_height, patch_width = self.model.embeddings.patch_embedding.kernel_size
        self.grid_size = (self.image_height // patch_height, self.image_width // patch_width)
        self.output_dim = output_dim

        # Transforms
        self.mean = self.processor.image_mean
        self.std = self.processor.image_std
        
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        # MLP
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(width, self.grid_size[0])).float()
        ).requires_grad_(False)
        self.attn_pool = nn.Sequential(
            nn.Linear(width, output_dim * 4, bias=True),
            ACT2FN["gelu"],
            nn.Linear(output_dim * 4, output_dim, bias=True)
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter((output_dim ** -0.5) * torch.randn(output_dim, output_dim), requires_grad=True)

    def forward_resampler(self, x):
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))
        x = x + pos_embed.unsqueeze(0)
        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def forward(self, x: torch.Tensor):
        # get feature
        x = self.model(x, output_hidden_states=True).hidden_states[-1]

        # resampler
        x = self.forward_resampler(x)
        return x

    def load_image(self, image_path, training):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        image = image.convert("RGB")

        image_tensor = self.image_transform(image)
        return image_tensor

    def encode(self, image_paths: List[str], training):
        images = []
        for image_path in image_paths:
            image = self.load_image(image_path, training)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        outputs = self.forward(images)
        return outputs
