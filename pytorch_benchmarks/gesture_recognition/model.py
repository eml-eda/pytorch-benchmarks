import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np 
from typing import Dict, Any, Optional

def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None
                        ) -> nn.Module:
    if model_name == 'bioformer':
        return Bioformer()
    else:
        raise ValueError(f"Unsupported model name {model_name}")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        bound1 = 1 / (dim ** .5)
        bound2 = 1 / (hidden_dim ** .5)
        nn.init.uniform_(self.net[0].weight, -bound1, bound1)
        nn.init.uniform_(self.net[0].bias, -bound1, bound1)
        nn.init.uniform_(self.net[3].weight, -bound2, bound2)
        nn.init.uniform_(self.net[3].bias, -bound2, bound2)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        bound = 1 / (dim ** .5)
        nn.init.uniform_(self.to_qkv.weight, -bound, bound)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        bound = 1 / (inner_dim ** .5)
        nn.init.uniform_(self.to_out[0].weight, -bound, bound)
        nn.init.uniform_(self.to_out[0].bias, -bound, bound)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Bioformer(nn.Module):
    def __init__(self, *, image_size = (1, 300), patch_size1 = (1, 3), patch_size2 = (1, 5), patch_size3 = (1, 0), ch_1 = 14, ch_2 = 32, ch_3 = None, tcn_layers = 2, num_classes = 8, dim_patch = 64, depth = 1, heads = 8, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0., use_cls_token=True, 
                 sessions="ignore", subjects="ignore", training_config="ignore", pretrained="ignore", chunk_idx="ignore", chunk_i="ignore"):
        super().__init__()
        num_patches = 30
        self.input_shape = (14,1,300)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        layerlist = []
        layerlist.append(nn.Conv2d(14, 16, kernel_size = (1,10), stride = (1,10))) 
        patch_dim = 16*30

        layerlist.append(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 30))
        layerlist.append(nn.Linear(patch_dim, dim_patch))
        self.to_patch_embedding = nn.Sequential(*layerlist)

        bound = 1 / (patch_dim ** .5)
        nn.init.uniform_(self.to_patch_embedding[-1].weight, -bound, bound)
        nn.init.uniform_(self.to_patch_embedding[-1].bias, -bound, bound)
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim_patch))
        else:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, dim_patch))
        nn.init.normal_(self.pos_embedding, mean=0, std=.02)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim_patch))
        nn.init.zeros_(self.cls_token)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim_patch, depth, heads, dim_head, dim_patch * 2, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_patch),
            nn.Linear(dim_patch, num_classes)
        )
        bound = 1 / (dim_patch ** .5)
        nn.init.uniform_(self.mlp_head[1].weight, -bound, bound)
        nn.init.uniform_(self.mlp_head[1].bias, -bound, bound)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else :
            x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x
