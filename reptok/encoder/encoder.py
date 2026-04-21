"""
Monolithic RepTok encoder/decoder definitions and loading helpers.

Reader map:
- Public release-facing API:
  - `RepTokImageEncoder`
  - `RepTokDecoderWrapper`
  - `load_raw`
- Internal sections:
  - Base AE / DINO backbone blocks
  - Projection + token utilities
  - Custom transformer / RF model definitions
  - Release wrappers and config-loading helpers

This file intentionally contains multiple components in one place. Start from the
public API above unless you need to modify the training/model internals.
"""

import hydra
import math
import numpy as np
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVTF

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from einops import rearrange, repeat
from functools import partial, reduce
from jaxtyping import Float
from omegaconf import OmegaConf
from pydoc import locate
from typing import (
    Any, Callable,
    List, Literal,
    MutableMapping,
    Optional, Sequence,
    Union,
)
from abc import ABC, abstractmethod

__all__ = ["RepTokImageEncoder", "RepTokDecoderWrapper", "load_raw"]

# ---------------------------------------------------------------------
# Base AE
# ---------------------------------------------------------------------

def expand_scaleshift(
    latent, # shape `B C ...`
    scaleshift, # shape `C`
):
    B, C, *DIMS = latent.shape
    assert C == scaleshift.shape[0], f"{C=} {scaleshift.shape}"
    repeat_dscr = "C -> 1 C "
    for _ in DIMS:
        repeat_dscr=f"{repeat_dscr}1 "
    return repeat(scaleshift, repeat_dscr).detach()
class AutoencoderKL(nn.Module):
    def __init__(
        self,
        scale: float | list[float] = 0.18215,
        shift: float | list[float] = 0.0,
        repo: str = "stabilityai/stable-diffusion-2-1",
        ae_cls: str = "diffusers.AutoencoderKL",
        ae_kwargs: dict[str, any] = { "subfolder": "vae" },
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        only_mode: bool =False,
    ):
        super().__init__()
        self.scale = scale if isinstance(scale, float) else nn.Parameter(torch.tensor(scale), requires_grad=False)
        self.shift = shift if isinstance(shift, float) else nn.Parameter(torch.tensor(shift), requires_grad=False)
        ae_cls_obj = locate(ae_cls)
        if ae_cls_obj is None:
            raise ImportError(f"Could not resolve autoencoder class {ae_cls!r}. Is the required package installed?")
        ae_kwargs = dict(ae_kwargs)
        if ae_kwargs.get("low_cpu_mem_usage") is None:
            ae_kwargs["low_cpu_mem_usage"] = False
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `local_dir_use_symlinks` argument is deprecated.*",
            )
            self.ae = ae_cls_obj.from_pretrained(repo, **ae_kwargs)
        self.ae.eval()
        self.ae.requires_grad_(False)
        if enable_slicing:
            self.ae.enable_slicing()
        if enable_tiling:
            self.ae.enable_tiling()
        self.only_mode = only_mode
    def forward(self, img):
        return self.encode(img)
    @torch.no_grad()
    def get_posterior(self, img):
        posterior = self.ae.encode(img, return_dict=False)[0]
        return posterior
    @torch.no_grad()
    def encode(self, img):
        posterior = self.ae.encode(img, return_dict=False)[0]
        if self.only_mode:
            latent = posterior.mode()
        else:
            latent = posterior.sample()
        shift = self.shift if isinstance(self.shift, float) else expand_scaleshift(latent, self.shift)
        scale = self.scale if isinstance(self.scale, float) else expand_scaleshift(latent, self.scale)
        latent = (latent - shift) * scale
        return latent
    @torch.no_grad()
    def decode(self, latent):
        shift = self.shift if isinstance(self.shift, float) else expand_scaleshift(latent, self.shift)
        scale = self.scale if isinstance(self.scale, float) else expand_scaleshift(latent, self.scale)
        latent = latent / scale + shift
        rec = self.ae.decode(latent, return_dict=False)[0]
        return rec
# ---------------------------------------------------------------------
# Base DINO
# ---------------------------------------------------------------------
def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
class LayerScale(nn.Module):
    def __init__(
        self,
        dim,
        init_values=1e-5,
        inplace=False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values)
    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer=Mlp,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx, sy = (
                float(w0 + self.interpolate_offset) / M,
                float(h0 + self.interpolate_offset) / M,
            )
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = (
            nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                # align_corners=False,
                **kwargs,
            )
            .permute(0, 2, 3, 1)
            .view(1, -1, dim)
        )
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(x.dtype)
    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.size(0), -1, -1), x[:, 1:]),
                dim=1,
            )
        return x
    def core_computation(self, x):
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return x_norm
    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        x_norm = self.core_computation(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }
    def forward(self, x, is_training=False, masks=None):
        ret = self.forward_features(x, masks)
        return ret if is_training else self.head(ret["x_norm_clstoken"])
def vit_small(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
def vit_base(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
def vit_large(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
def vit_giant2(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=8 / 3,
        num_register_tokens=num_register_tokens,
        ffn_layer=SwiGLUFFN,
        **kwargs,
    )
# Variants with registers based on `https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py`.
# Note the interpolate args for positional encoding are different to variants without registers.
vit_small_reg = partial(
    vit_small,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)
vit_base_reg = partial(
    vit_base,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)
vit_large_reg = partial(
    vit_large,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)
vit_giant2_reg = partial(
    vit_giant2,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)

###
# DINO Mods
###
PARTIAL_REG_KEY_PREFIX = "x_norm_split_reg"
GLOBAL_KEY = "x_norm_global"
NO_FILTER_KEY = "x_norm_all"
def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module], access_string: str):
    # taken from https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
    """Retrieve a module nested in another by its access string.
    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
def better_resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    ss = imgs.shape
    assert ss[-3] == 3, f"{imgs.shape}, {image_size}"
    H, W = ss[-2:]
    if len(ss) == 3:
        imgs = imgs.unsqueeze(0)
    side = min(H, W)
    factor = side // image_size
    imgs = TVTF.center_crop(imgs, [side, side])
    if factor > 1:
        imgs = F.avg_pool2d(imgs, factor)
    imgs = F.interpolate(imgs, [image_size, image_size], mode="bilinear")
    if len(ss) == 3:
        imgs = imgs[0]
    return imgs
class ConditionEncoderBase(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
    def forward(self, x, x_latent, **kwargs) -> torch.Tensor:
        pass
class RepTokEncoder(ConditionEncoderBase):
    def __init__(
        self,
        dino: nn.Module,
    ) -> None:
        super().__init__()
        self.dino = dino
    def forward(self, x, x_latent, **kwargs) -> torch.Tensor:
        dino_output = self.dino(x)
        if isinstance(dino_output, tuple):
            output, loss = dino_output
            self.loss = loss
            dino_output = output
        return dino_output
class DINOFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_size: int = 448,
        base_url: str = "facebookresearch/dinov2",
        model_version: str = "dinov2_vitl14_reg",
        impl_call: str = None,
        from_local: bool = False,
        use_pretrained: bool = True,
        use_frozen_target: bool = False,
        target_key: str = "x_norm_patchtokens",
        normalize_mean: float | tuple[float, float, float] = [0.485, 0.456, 0.406],
        normalize_std: float | tuple[float, float, float] = [0.229, 0.224, 0.225],
        gradient_last_blocks: None | int = None,
        reshape: bool = True,
        requires_grad: bool = False,
        compile: bool = True,
        unfreeze: list[str] = [],
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.model_size = model_size
        self.reshape = reshape
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is available.*")
            model: nn.Module = torch.hub.load(base_url, model_version, source="local" if from_local else "github")
        if impl_call is not None:
            self.model = locate(impl_call)()
            if use_pretrained:
                self.model.load_state_dict(model.state_dict())
            if use_frozen_target:
                self.target_model = locate(impl_call)()
                self.target_model.load_state_dict(model.state_dict())
                self.target_model.requires_grad_(False)
                self.target_model.eval()
                if compile:
                    if hasattr(self.target_model, "core_computation"):
                        self.target_model.core_computation = torch.compile(
                            self.target_model.core_computation, fullgraph=True, dynamic=False, mode="reduce-overhead"
                    )
                    else:
                        self.forward = torch.compile(self.forward, dynamic=False, mode="reduce-overhead")
        else:
            self.model = model
        if requires_grad:
            self.model.requires_grad_(True)
            self.model.train()
        else:
            self.model.requires_grad_(False)
            self.model.eval()
        self.target_key = target_key
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.gradient_last_blocks = gradient_last_blocks
        if gradient_last_blocks is not None and gradient_last_blocks > 0:
            for l in self.model.blocks[-gradient_last_blocks:]:
                l.requires_grad_(True)
                l.train()
        for submodule_name in unfreeze:
            submodule = get_module_by_name(self.model, submodule_name)
            submodule.requires_grad_(True)
        if compile:
            if hasattr(self.model, "core_computation"):
                self.model.core_computation = torch.compile(
                    self.model.core_computation, fullgraph=True, dynamic=False, mode="reduce-overhead"
                )
            else:
                self.forward = torch.compile(self.forward, dynamic=False, mode="reduce-overhead")
    def forward(
        self, imgs: Float[torch.Tensor, "B C H W"]
    ) -> Float[torch.Tensor, "B D h' w'"] | Float[torch.Tensor, "B D N"] | dict[str, torch.Tensor]:
        assert imgs.min() >= -1.0, f"out of bounds {imgs.min()} {imgs.max()}"
        assert imgs.max() <= 1.0, f"out of bounds {imgs.min()} {imgs.max()}"
        assert len(imgs.shape) == 4
        imgs = better_resize(imgs, self.model_size)
        B, _, H, W = imgs.shape
        imgs = (imgs + 1.0) / 2.0  # to 0-1 range
        # copied from transformers preprocessor
        imgs = TVTF.normalize(imgs, self.normalize_mean, self.normalize_std)
        feats:dict[str, torch.Tensor] = self.model.forward_features(imgs.clone())
        cls_token = feats["x_norm_clstoken"]
        has_target = hasattr(self, "target_model") and self.training
        if has_target:
            target_feats = self.target_model.forward_features(imgs)
            target_cls = target_feats["x_norm_clstoken"]
            cossim = 1 - torch.nn.functional.cosine_similarity(cls_token.float(), target_cls.float(), dim=-1)
        # if hasattr(self, "target_model") and not self.training:
        #     target_feats = self.target_model.forward_features(imgs)
        #     target_cls = target_feats["x_norm_clstoken"]
        #     og_norm = torch.linalg.norm(target_cls)
        #     new_norm = torch.linalg.norm(cls_token.clone())
        #     print(f"DINO cls norms: og={og_norm.mean().item()}|{og_norm.std().item()} new={new_norm.mean().item()}|{new_norm.std().item()}")
        cls_token = rearrange(cls_token, "B D -> B 1 D") # just to ensure all tokens have the form B N D
        feats["x_norm_clstoken"] = cls_token
        if self.target_key in feats.keys():
            result = feats[self.target_key]
        elif self.target_key == NO_FILTER_KEY:
            result = feats
        elif self.target_key == GLOBAL_KEY:
            D = cls_token.shape[-1]
            reg_tokens = feats.get("x_norm_regtokens", torch.zeros((B, 0, D), device=imgs.device, dtype=imgs.dtype))
            global_tokens = torch.cat(
                [
                    cls_token,
                    reg_tokens,
                ],
                dim=-2,
            )
            result = global_tokens
        elif self.target_key.startswith(PARTIAL_REG_KEY_PREFIX):
            nr_split = int(self.target_key[len(PARTIAL_REG_KEY_PREFIX):])
            assert nr_split > 0, "Need to split at least one register token."
            D = feats["x_norm_clstoken"].shape[-1]
            reg_tokens = feats.get("x_norm_regtokens", torch.zeros((B, 0, D), device=imgs.device, dtype=imgs.dtype))
            split_reg_tokens = reg_tokens[:, :nr_split]
            result = split_reg_tokens
        else:
            raise ValueError(f"unsupported target_key `{self.target_key}`")
        if has_target:
            return result, cossim
        return result
def flatten_vid(vid: Float[torch.Tensor, "B C T H W"]) -> tuple[Float[torch.Tensor, "(B T) C H W"], int, int]:
    B, _, T, _, _ = vid.shape
    imgs = rearrange(vid, "B C T H W -> (B T) C H W")
    return imgs, B, T
def unflatten_vid(imgs:Float[torch.Tensor, "(B T) C H W"], B:int, T:int) -> Float[torch.Tensor, "B C T H W"]:
    vid = rearrange(imgs, "(B T) C H W -> B C T H W", B=B, T=T)
    return vid
def unflatten_seq(seq:Float[torch.Tensor, "(B T) N D"], B:int, T:int) -> Float[torch.Tensor, "B T N D"]:
    res = rearrange(seq, "(B T) N D -> B T N D", B=B, T=T)
    return res
def reflatten_seq(seq:Float[torch.Tensor, "B T N D"],) -> Float[torch.Tensor, "B (T N) D"]:
    res = rearrange(seq, "B T N D -> B (T N) D",)
    return res
class MultiFrameDINOEncoder(DINOFeatureEncoder):
    def __init__(self, nr_frames:int, embed_dim:int, use_reflatten:bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nr_frames = nr_frames
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, nr_frames, 1, embed_dim))
        self.use_reflatten = use_reflatten
    def reform_seq(self, seq:Float[torch.Tensor, "(B T) N D"], B:int, T:int):
        seq:Float[torch.Tensor, "B T N D"] = unflatten_seq(seq, B, T)
        seq = seq + self.temp_pos_embed  # this won't work if T and self.nr_frames don't match...
        if self.use_reflatten:
            seq = reflatten_seq(seq)
        return seq
    def forward(
        self, vid: Float[torch.Tensor, "B C T H W"],
    ):
        imgs, B, T = flatten_vid(vid)
        assert T == self.nr_frames, f"expected {T=} and {self.nr_frames} to be equal"
        feats = super().forward(imgs)
        has_loss = isinstance(feats, tuple)
        if has_loss:
            f, loss = feats
            feats = f
        if isinstance(feats, torch.Tensor):
            result = self.reform_seq(feats, B, T)
        elif isinstance(feats, dict):
            result = {
                k: self.reform_seq(v, B, T)
                for k, v in feats.items()
                if k.startswith("x_norm_")
            } # filter out the masks
        else:
            raise ValueError(f"Unsupported return type of DINO: {type(feats)}")
        if has_loss:
            return result, loss
        return result
class RepTokConditionEncoder(MultiFrameDINOEncoder):
    def __init__(
        self,
        backbone:nn.Module,
        nr_new_reg:int,
        embed_dim:int,
        new_target_key:str = "only_reg",
        *args,
        **kwargs,
    ):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.backbone = backbone
        self.nr_new_reg = nr_new_reg
        self.new_reg = nn.Parameter(torch.randn(1, nr_new_reg, embed_dim))
        self.new_target_key = new_target_key
        # assert self.target_key == NO_FILTER_KEY
    def forward(self, vid: Float[torch.Tensor, "B C T H W"],) -> dict[str, torch.Tensor]:
        feats = super().forward(vid)
        has_loss = isinstance(feats, tuple)
        if has_loss:
            f, loss = feats
            feats = f
        B = vid.shape[0]
        T = vid.shape[2]
        new_reg_tokens = repeat(
            self.new_reg,
            "1 N D -> B N D",
            # "1 N D -> B 1 N D" if self.use_reflatten else "1 N D -> B N D",
            B=B,
        )
        all_tokens = [
            new_reg_tokens,
        ]
        if self.target_key == "x_norm_all":
            cls_tokens = feats["x_norm_clstoken"]
            reg_tokens = feats["x_norm_regtokens"]
            patch_tokens = feats["x_norm_patchtokens"]
            og_tokens = [
                cls_tokens,
                reg_tokens,
                patch_tokens,
            ]
            if not self.use_reflatten:
                N_og = reduce(lambda x1,x2 : x1 + x2, [tokens.shape[-2] for tokens in og_tokens])
                og_tokens = torch.cat(og_tokens, dim=-2)
            all_tokens.extend(og_tokens)
        else:
            if not self.use_reflatten:
                N_og = feats.shape[-2]
                feats = rearrange(feats, "B ... D -> B (...) D")
            all_tokens.append(feats)
        recat = torch.cat(
            all_tokens,
            dim=-2,
        )
        N = recat.shape[1]
        pos = torch.zeros((B, 1, N,), device=recat.device, dtype=recat.dtype)
        recat = rearrange(recat, "B ... D -> B D ...")
        new_feats = self.backbone(recat, pos)
        new_feats = rearrange(new_feats, "B D ... -> B ... D")
        if self.new_target_key == "only_reg":
            result = new_feats[:, :self.nr_new_reg]
        elif self.new_target_key == "only_nonreg":
            result = new_feats[:, self.nr_new_reg:]
            if not self.use_reflatten:
                result = rearrange(result, "B (T N) D -> B T N D", T=T, N=N_og)
        elif self.new_target_key == "all":
            result = new_feats
        else:
            raise ValueError(f"unsupported {self.new_target_key=}")
        if has_loss:
            return result, loss
        return result

###
# ZipTok minimal proj
###

def pad_to(x, out_dim):
    diff = out_dim - x.shape[-1]
    if diff <= 0:
        return x
    return torch.cat([x, torch.zeros(*x.shape[:-1], diff, device=x.device, dtype=x.dtype)], dim=-1)
class Projection(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
    def forward(self, x):
        return self.ln(self.fc(x))
class ZeroInitProjection(Projection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = zero_init(self.fc)
class ResidualProjection(Projection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = zero_init(self.fc)
    def forward(self, x):
        skip = x
        x = super().forward(x)
        return x + pad_to(skip, x.shape[-1])  # pad to match the output shape
class ResidualProjectionBackbone(ResidualProjection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "B D N -> B N D")
        x = super().forward(x)
        x = rearrange(x, "B N D -> B D N")
        return x
###
# RoPE
###

def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2
def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)
def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj
    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj
    return y_min, y_max, x_min, x_max
def make_axial_pos_2d(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2

    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)
class AbstractPosEnc(nn.Module, ABC):
    def __init__(self, d_head, n_heads):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
    @abstractmethod
    def forward(self, pos):
        pass
    @abstractmethod
    def apply_emb(self, x, theta):
        pass
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    result = torch.cat((y1, y2, x3), dim=-1)
    return result
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1], f"{d=} x={x.shape} theta={theta.shape}"
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)
class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x
    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj
    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None
def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)
class AxialRoPEBase(AbstractPosEnc):
    def __init__(self, d_head, n_heads, in_place=False, skip_first_n: int = 0):
        super().__init__(d_head, n_heads)
        self.in_place = in_place
        self.skip_first_n = skip_first_n
    def apply_emb(self, x, theta):
        skip = x[..., : self.skip_first_n, :]
        x = x[..., self.skip_first_n :, :]
        if self.in_place:
            result = apply_rotary_emb_(x, theta)
        else:
            result = apply_rotary_emb(x, theta)
        x = torch.cat([skip, result], dim=-2)
        return x
    @abstractmethod
    def forward(self, pos):
        pass
class AxialRoPE2D(AxialRoPEBase):
    def __init__(self, dim, n_heads, learnable_freqs=False, relative_canvas=True, skip_first_n=0, half_embedding: bool = True,):
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim = dim // 2
        super().__init__(dim, n_heads, in_place=not learnable_freqs, skip_first_n=skip_first_n)
        self.learnable_freqs = learnable_freqs

        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.stack([torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freqs.view(2, dim // 4, n_heads).mT.contiguous(), requires_grad=learnable_freqs)
    def extra_repr(self):
        return f"dim={self.freqs.shape[-2] * 4}, n_heads={self.freqs.shape[-1]}"
    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)

###
# Transformer Utils
###
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    result = (q * scale_q.to(q.dtype), k * scale_k.to(k.dtype))
    del scale_k, scale_q, sqrt_scale, sum_sq_k, sum_sq_q
    return result
def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer
def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param
def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module
def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    x_conv = x.to(dtype)
    x_square = x_conv**2
    mean_sq = torch.mean(x_square, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    result = x * scale.to(x.dtype)
    del scale, mean_sq, x_square, x_conv
    return result
def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)
class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features
    def forward(self, x):
        return linear_swiglu(x, self.weight, self.bias)
class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(nn.Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")
    def extra_repr(self):
        return f"eps={self.eps},"
    def forward(self, x, cond):
        # removed one additional expansion here
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)
class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))
    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"
    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm=None, dropout=0.0):
        super().__init__()
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearSwiGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))
    @torch.compile(fullgraph=True)
    def forward(
        self,
        x,
        # check_dict=None,
        # cond_norm=None,
        **kwargs,
    ):
        cond_norm = kwargs.get("cond_norm", None)
        check_dict = kwargs.get("check_dict", {})
        # def forward(self, x, check_dict, cond_norm=None, **kwargs):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
            check_dict["cond_norm"] = True
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip
class SimpleProj(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = apply_wd(nn.Linear(in_features, out_features, bias=False))

    def forward(self, x, **kwargs):
        return self.proj(x)
def CustomProj(
    type: Literal["split", "split_last" "merge"],
    in_features,
    out_features,
    in_cls,
    out_cls,
    in_params={},
    out_params={},
):
    if type == "split" or type == "split_last":
        return locate(out_cls)(in_features, out_features, **out_params)
    elif type == "merge":
        return locate(in_cls)(in_features, out_features, **in_params)
    else:
        raise ValueError(f"Unknown type: {type}")
class AddTokensProjv2(nn.Module):
    def __init__(self, in_features, out_features, cat_dim: int = 1, pre_proj_cls=None, pre_proj_params={}, flatten: bool = False, add_pos: bool = False):
        super().__init__()
        self.proj = nn.Linear(out_features, out_features, bias=False)
        self.pre_proj = locate(pre_proj_cls)(in_features, out_features, **pre_proj_params) if pre_proj_cls else None
        self.cat_dim = cat_dim
        self.flatten = flatten
        self.add_pos = add_pos
    def forward(
        self,
        x: Float[torch.Tensor, "b ... d"],
        pos: Float[torch.Tensor, "b ... n"],
        x_extra: Float[torch.Tensor, "b ... d"],
        pos_extra: Float[torch.Tensor, "b ... n"] | None = None,
        **kwargs,
    ):
        if "check_dict" in kwargs:
            kwargs["check_dict"]["x_extra"] = True
            kwargs["check_dict"]["pos_extra"] = True
        assert self.add_pos == (pos_extra is not None), f"{self.add_pos=} {pos_extra is not None}"
        if self.pre_proj:
            x, pos = self.pre_proj(x, pos, **kwargs)
        if self.flatten:
            x = rearrange(x, "b ... c -> b (...) c")
            if self.add_pos:
                pos = rearrange(pos, "b ... c -> b (...) c")
        # print(f"add token x={x.shape} x_extra={x_extra.shape}")
        x = self.proj(
            torch.cat([x_extra, x], dim=self.cat_dim),
        )
        if self.add_pos:
            pos = torch.cat([pos_extra, pos], dim=self.cat_dim)
        return x, pos
class RemoveTokensProjv2(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rm_dim: int = 1,
        rm_first: int | None = None,
        rm_last: int | None = None,
        unflatten: dict[str, int] | None = None,
        rm_pos: bool = False,
        post_proj_cls=None,
        post_proj_params={},
    ):
        super().__init__()
        self.rm_dim = rm_dim
        self.rm_first = rm_first
        self.rm_last = rm_last
        self.rm_pos = rm_pos
        self.post_proj = locate(post_proj_cls)(in_features, out_features, **post_proj_params) if post_proj_cls else None
        self.unflatten = unflatten
        if self.unflatten is not None:
            self.unflatten_dscr = ""
            for key in self.unflatten.keys():
                self.unflatten_dscr = f"{self.unflatten_dscr} {key}"
    def forward(self, x, **kwargs):
        pos = kwargs.get("pos", None)
        if self.rm_first is not None:
            assert self.rm_dim == 1, "Not implemented for rm_dim != 1"
            x = x[:, self.rm_first :]
            pos = pos[:, self.rm_first :] if not pos is None and self.rm_pos else pos
        if self.rm_last is not None:
            assert self.rm_dim == 1, "Not implemented for rm_dim != 1"
            x = x[:, : -self.rm_last]
            pos = pos[:, : -self.rm_last] if not pos is None and self.rm_pos else pos
        if self.unflatten is not None:
            x = rearrange(x, f"b ({self.unflatten_dscr}) c -> b {self.unflatten_dscr} c", **self.unflatten)
        if self.post_proj:
            x = self.post_proj(x, **kwargs) if pos is None else self.post_proj(x, pos, **kwargs)
        if pos is None:
            return x
        return x, pos
def Patch2D(type: Literal["split", "split_last" "merge"], in_features, out_features, patch_size=(2, 2)):
    if type == "split":
        return TokenSplit2D(in_features, out_features, patch_size)
    if type == "split_last":
        return TokenSplitLast2D(in_features, out_features, patch_size)
    elif type == "merge":
        return TokenMerge2D(in_features, out_features, patch_size)
    else:
        raise ValueError(f"Unknown type: {type}")
class TokenMerge2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(nn.Linear(in_features * self.h * self.w, out_features, bias=False))
    def forward(self, x, pos, **kwargs):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)

        pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=self.h, nw=self.w)
        return self.proj(x), torch.mean(pos, dim=-2)
class TokenSplit2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(nn.Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)
    def forward(self, x, skip=None, **kwargs):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        if skip is None:
            return x
        return torch.lerp(skip, x, self.fac.to(x.dtype))
# Doens't have skip but norm
class TokenSplitLast2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.norm = RMSNorm(in_features)
        self.proj = apply_wd(nn.Linear(in_features, out_features * self.h * self.w, bias=False))
        nn.init.zeros_(self.proj.weight)
    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return x
def prep_rearrange(dims_dscr: str, patch_size: Sequence[int]):
    entangled = ""
    patch_factors = ""
    refactored = ""
    args = {}
    for dim, factor in zip(dims_dscr.split(" "), patch_size):
        entangled = "{} ({} n{})".format(entangled, dim, dim)
        patch_factors = "{} n{}".format(patch_factors, dim)
        refactored = "{} {}".format(refactored, dim)
        args["n{}".format(dim)] = factor
    entangled = entangled.strip()
    patch_factors = patch_factors.strip()
    refactored = refactored.strip()
    return entangled, patch_factors, refactored, args
def PatchXD(
    type: Literal["split", "split_last", "merge"],
    in_features,
    out_features,
    patch_size: Sequence[int],
    dims_dscr: str,
    **kwargs,
):
    if type == "split":
        return TokenSplitXD(in_features, out_features, patch_size, dims_dscr, use_skip=True, **kwargs)
    if type == "split_last":
        return TokenSplitXD(in_features, out_features, patch_size, dims_dscr, use_skip=False, **kwargs)
    elif type == "merge":
        return TokenMergeXD(in_features, out_features, patch_size, dims_dscr, **kwargs)
    else:
        raise ValueError(f"Unknown type: {type}")
class TokenMergeXD(nn.Module):
    def __init__(
        self, in_features, out_features, patch_size, dims_dscr, init_zero_proj=False, cond_features=0,
    ):
        super().__init__()
        self.patch_size = patch_size
        entangled, patch_factors, refactored, args = prep_rearrange(dims_dscr, patch_size)
        self.rearrange_x_str = f"... {entangled} e -> ... {refactored} ({patch_factors} e)"
        self.rearrange_pos_str = f"... {entangled} e -> ... {refactored} ({patch_factors}) e"
        self.rearrange_args = args
        self.cond_features = cond_features if cond_features is not None else 0
        self.proj = apply_wd(
            nn.Linear(in_features * math.prod(patch_size) + self.cond_features, out_features, bias=False)
        )
        if init_zero_proj:
            nn.init.zeros_(self.proj.weight)

    def forward(self, x, pos, cond_tokens=None, check_dict: dict = None, **kwargs):
        x = rearrange(x, self.rearrange_x_str, **self.rearrange_args)
        pos = rearrange(pos, self.rearrange_pos_str, **self.rearrange_args)
        if cond_tokens is not None and self.cond_features > 0:
            check_dict["cond_tokens"] = True
            x = torch.cat((x, cond_tokens), dim=-1)
        x = self.proj(x)
        pos = torch.mean(pos, dim=-2)
        return x, pos
class TokenSplitXD(nn.Module):
    def __init__(
        self, in_features, out_features, patch_size, dims_dscr, init_zero_proj=False, use_skip=True, cond_features=0
    ):
        super().__init__()
        self.patch_size = patch_size
        entangled, patch_factors, refactored, args = prep_rearrange(dims_dscr, patch_size)
        self.rearrange_str = f"... {refactored} ({patch_factors} e) -> ... {entangled} e"
        self.rearrange_args = args
        self.cond_features = cond_features if cond_features is not None else 0
        self.proj = apply_wd(
            nn.Linear(in_features + self.cond_features, out_features * math.prod(patch_size), bias=False)
        )
        if init_zero_proj:
            nn.init.zeros_(self.proj.weight)
        self.use_skip = use_skip
        if use_skip:
            self.fac = nn.Parameter(torch.ones(1) * 0.5)
        else:
            self.norm = RMSNorm(in_features)
    def forward(self, x, skip=None, cond_tokens=None, check_dict: dict = None, **kwargs):
        if not self.use_skip:
            x = self.norm(x)
        if cond_tokens is not None and self.cond_features > 0:
            check_dict["cond_tokens"] = True
            x = torch.cat((x, cond_tokens), dim=-1)
        x = self.proj(x)
        x = rearrange(x, self.rearrange_str, **self.rearrange_args)
        if not self.use_skip or skip is None:
            return x
        return torch.lerp(skip, x, self.fac.to(x.dtype))
@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float
class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearSwiGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))
    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip
class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)
    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
####
# Basic Attention layer/block
####
class GenericTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        pos_enc_cls,
        pos_enc_params={},
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
    ):
        super().__init__()
        d_ff = d_model * ff_expand
        self.self_attn = GenericAttentionBlock(d_model, pos_enc_cls, pos_enc_params, d_head, d_cond_norm, dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)
    def forward(self, x, pos, **kwargs):
        x = self.self_attn(x, pos, **kwargs)
        x = self.ff(x, **kwargs)
        return x
class GenericAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_cls,
        pos_enc_params,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        compile: bool = False,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.qkv_proj = apply_wd(nn.Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = locate(pos_enc_cls)(d_head, self.n_heads, **pos_enc_params)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(nn.Linear(d_model, d_model, bias=False)))
        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True)
    def extra_repr(self):
        return f"d_head={self.d_head},"
    def forward(self, x, pos, **kwargs):
        check_dict = kwargs.get("check_dict", {})
        cond_norm = kwargs.get("cond_norm", None)
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
            check_dict["cond_norm"] = True
        else:
            x = self.norm(x)
        qkv = self.qkv_proj(x)
        if pos is not None:
            pos = pos.to(qkv.dtype)
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        if pos is not None:
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip
####
# Transformer / DiT backbone
####
class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x
class OutputCallbackLevel(Level):
    def __init__(self, layers, callback: Callable[[int, Any], None]):
        super().__init__(layers)
        self.callback = callback
    def forward(self, x, *args, **kwargs):
        for i, layer in enumerate(self):
            x = layer(x, *args, **kwargs)
            if not self.callback is None:
                x = self.callback(i, x, **kwargs)
        return x
class Transformer(nn.Module):
    def __init__(self, in_features, out_features, main_level=None, up_levels=[], down_levels=[], main_level_cls=Level):
        super().__init__()
        # assumes up and down levels are already in the correct order
        prev_in = in_features
        prev_out = out_features
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        self.merges, self.splits = nn.ModuleList(), nn.ModuleList()
        # high res -> low res
        for i, level in enumerate(down_levels):
            width = level.width
            self.merges.append(
                locate(level.proj_cls)(type="merge", in_features=prev_in, out_features=width, **level.proj_params)
            )
            layer_factory = lambda: locate(level.layer_class)(d_model=width, **level.layer_params)
            self.down_levels.append(Level([layer_factory() for _ in range(level.depth)]))
            prev_in = width
        # NOTE: up levels in same order as down levels
        # so high res -> low res
        for i, level in enumerate(up_levels):
            width = level.width
            layer_factory = lambda: locate(level.layer_class)(d_model=width, **level.layer_params)
            self.up_levels.append(Level([layer_factory() for _ in range(level.depth)]))
            self.splits.append(
                locate(level.proj_cls)(
                    type="split" if i + 1 < len(up_levels) else "split_last",
                    in_features=width,
                    out_features=prev_out,
                    **level.proj_params,
                )
            )
            prev_out = width
        self.mid_level, self.mid_merge, self.mid_split = None, None, None
        if main_level is not None:
            width = main_level.width
            layer_factory = lambda: locate(main_level.layer_class)(d_model=main_level.width, **main_level.layer_params)
            self.mid_level = main_level_cls([layer_factory() for _ in range(main_level.depth)])
            self.mid_merge = locate(main_level.proj_cls)(
                type="merge", in_features=prev_in, out_features=main_level.width, **main_level.proj_params
            )
            self.mid_split = locate(main_level.proj_cls)(
                type="split" if len(up_levels) > 0 else "split_last",
                in_features=main_level.width,
                out_features=prev_out,
                **main_level.proj_params,
            )
    def forward(self, x: Float[torch.Tensor, "B C *DIMS"], pos: Float[torch.Tensor, "B cn *DIM"], **kwargs):
        check_dict = {k: False for k in kwargs.keys()}
        x = rearrange(x, "b c ... -> b ... c")
        pos = rearrange(pos, "b cn ... -> b ... cn")
        C_pos = pos.shape[-1]
        skips, poses = [], []
        for merge, level in zip(self.merges, self.down_levels):
            skips.append(x)
            x, pos = merge(x, pos, check_dict=check_dict, **kwargs)
            poses.append(pos)
            B, *DIMS, C = x.shape
            kwargs = { "x_shape_orig": x.shape } | kwargs
            check_dict["x_shape_orig"] = True
            x = x.reshape(B, -1, C)
            pos = pos.reshape(B, -1, C_pos)
            x = level(x, pos=pos, check_dict=check_dict, **kwargs)
            x = x.reshape(B, *DIMS, C)
            pos = pos.reshape(B, *DIMS, C_pos)
        if self.mid_level is not None:
            skip = x
            x, pos = self.mid_merge(x, pos, check_dict=check_dict, **kwargs)
            B, *X_DIMS, C = x.shape
            _, *POS_DIMS, _ = pos.shape
            kwargs = { "x_shape_orig": x.shape } | kwargs
            check_dict["x_shape_orig"] = True
            x = x.reshape(B, -1, C)
            pos = pos.reshape(B, -1, C_pos)
            x = self.mid_level(x, pos=pos, check_dict=check_dict, **kwargs)
            x = x.reshape(B, *X_DIMS, C)
            pos = pos.reshape(B, *POS_DIMS, C_pos)
            x = self.mid_split(x, skip=skip, check_dict=check_dict, **kwargs)
        for split, level, skip, pos in reversed(list(zip(self.splits, self.up_levels, skips, poses))):
            B, *DIMS, C = x.shape
            kwargs = { "x_shape_orig": x.shape } | kwargs
            check_dict["x_shape_orig"] = True
            x = x.reshape(B, -1, C)
            pos = pos.reshape(B, -1, C_pos)
            x = level(x, pos=pos, check_dict=check_dict, **kwargs)
            x = x.reshape(B, *DIMS, C)
            pos = pos.reshape(B, *DIMS, C_pos)
            x = split(x, skip=skip, check_dict=check_dict, **kwargs)
        x = rearrange(x, "b ... c -> b c ...")
        assert all([v for v in check_dict.values()]), f"Not all kwargs were used in the forward pass: {check_dict}"
        return x
    
####
# RF base
####

class RF(nn.Module, ABC):
    def __init__(
        self,
        backbone: nn.Module,
        mapping,
        train_timestep_sampling: Literal["logit_sigmoid", "uniform"] = "logit_sigmoid",
        time_cond_type: Literal["sigma", "rf_t"] = "rf_t",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.train_timestep_sampling = train_timestep_sampling
        self.mapping = mapping

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.time_cond_type = time_cond_type
        self.mapping = tag_module(
            MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping"
        )
        self.mapping.compile()
    @abstractmethod
    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        pass
    def get_conditioning(self, t: Float[torch.Tensor, "b"], **kwargs) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond_time = self.mapping(time_emb)
        return {"cond_norm": cond_time}
    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs) -> Float[torch.Tensor, "b"]:
        B = x.size(0)
        if self.train_timestep_sampling == "logit_sigmoid":
            t = torch.sigmoid(torch.randn((B,), device=x.device))
        elif self.train_timestep_sampling == "uniform":
            t = torch.rand((B,), device=x.device)
        else:
            raise ValueError(f'Unknown train timestep sampling method "{self.train_timestep_sampling}".')
        texp = t.view([B, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        # make t, zt into same dtype as x
        dtype = x.dtype
        zt, t = zt.to(dtype), t.to(dtype)
        cond_dict = self.get_conditioning(t, **data_kwargs)
        pos = self.get_pos(zt)
        v = z1 - x
        vtheta = self.backbone(zt, pos=pos, **cond_dict)
        return ((v - vtheta) ** 2).mean(dim=list(range(1, len(vtheta.shape))))
    @torch.no_grad()
    def sample(
        self, z: Float[torch.Tensor, "b c ..."], sample_steps=50, **data_kwargs
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        B, C, *DIMS = z.shape
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * B, device=z.device, dtype=z.dtype).view([B, *([1] * len(z.shape[1:]))])
        pos = self.get_pos(z)
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * B, device=z.device, dtype=z.dtype)
            cond_dict = self.get_conditioning(t, **data_kwargs)
            vc_cond = self.backbone(z, pos=pos, **cond_dict)
            z = z - dt * vc_cond
        return z
class LatentRF2D(RF):
    def __init__(
        self,
        ae: AutoencoderKL,
        val_shape: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ae = ae
        self.val_shape = val_shape
    def forward(self, x: Float[torch.Tensor, "b ..."], x_latent: torch.Tensor | None=None, **data_kwargs) -> Float[torch.Tensor, "b"]:
        if x_latent is None:
            latent = self.ae.encode(x)
        else:
            mean, logvar = torch.chunk(x_latent, 2, dim=1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            latent = mean + std * torch.randn(mean.shape, device=std.device)
            latent = (latent + self.ae.shift) * self.ae.scale
            latent = latent.to(dtype=x.dtype)
        return super().forward(latent, **data_kwargs)
    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        B, _, *DIMS = x.shape
        pos = make_axial_pos_2d(*DIMS, device=x.device).view(1, *DIMS, -1).expand(B, -1, -1, -1)
        return pos.movedim(-1, 1)
    def sample(
        self, z: Float[torch.Tensor, "b c ..."], sample_steps=50, return_list: bool = False, **data_kwargs
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        latent = super().sample(z, sample_steps=sample_steps, return_list=return_list, **data_kwargs)
        return self.ae.decode(latent)
    
####
# UnDINO base
####

class RepTokLatentRF2D(LatentRF2D):
    def __init__(
        self,
        encoder: nn.Module,
        visualization_fps: int = 25,
        use_pos_extra: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.visualization_fps = visualization_fps
        self.use_pos_extra = use_pos_extra
    def get_conditioning(
        self, t: Float[torch.Tensor, "b"], x_cond: Float[torch.Tensor, "b c h w"], **kwargs
    ) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond_time = self.mapping(time_emb)
        x_cond_vid = repeat(x_cond, "b c h w -> b c 1 h w")
        latent = self.encoder(x_cond_vid, x_latent=None, **kwargs)
        if self.use_pos_extra:
            B, T, N, D = latent.shape
            pos_extra = make_axial_pos_2d(1, 1, device=latent.device)
            pos_extra = repeat(pos_extra, " 1 D -> B N D", B=B, N=N)
            pos_extra = rearrange(pos_extra, "B ... D -> B (...) D")
            latent = rearrange(latent, "B ... D -> B (...) D")
        result = {
            "cond_norm": cond_time,
            "x_extra": latent,
        }
        if self.use_pos_extra:
            result["pos_extra"] = pos_extra
        return result
    def forward(self, x: Float[torch.Tensor, "b c h w"], **data_kwargs) -> Float[torch.Tensor, "b"]:
        rf_loss = super().forward(x=x, x_cond=x.clone(), **data_kwargs)
        loss = {
            "rf": rf_loss,
        }
        if hasattr(self.encoder, "loss"):
            loss["encoder"] = self.encoder.loss
            delattr(self.encoder, "loss")
        return loss

# ---------------------------------------------------------------------
# Release-facing wrappers
# ---------------------------------------------------------------------

class RepTokDecoderWrapper(nn.Module):
    """Inference wrapper around a trained `RepTokLatentRF2D` decoder.

    It converts a DINO cls-token `(B, D)` into the decoder's conditioning inputs
    and runs iterative latent RF sampling before decoding with the internal AE.
    """
    def __init__(self, decoder:RepTokLatentRF2D):
        super().__init__()
        self.decoder = decoder
    def get_time_cond(self, t):
        if self.decoder.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.decoder.time_cond_type == "rf_t":
            c_noise = t
        time_emb = self.decoder.time_in_proj(self.decoder.time_emb(c_noise[..., None]))
        cond_time = self.decoder.mapping(time_emb)
        return cond_time
    def get_conditioning(
        self,
        cls_token,  # shape B x D
    ):
        nr_frames = 1
        # DINOFeatureEncoder
        feats = repeat(cls_token, "B D -> B 1 D",)
        # MultiFrameDINOEncoder
        B = cls_token.shape[0]
        dino: RepTokConditionEncoder = self.decoder.encoder.dino
        feats = dino.reform_seq(feats, B, nr_frames)
        # RepTokConditionEncoder
        new_reg_tokens = repeat(
            dino.new_reg,
            "1 N D -> B N D",
            B=B,
        )
        all_tokens = [
            new_reg_tokens,
        ]
        if not dino.use_reflatten:
            N_og = feats.shape[-2]
            feats = rearrange(feats, "B ... D -> B (...) D")
        all_tokens.append(feats)
        recat = torch.cat(
            all_tokens,
            dim=-2,
        )
        N = recat.shape[1]
        pos = torch.zeros((B, 1, N,), device=recat.device, dtype=recat.dtype)
        recat = rearrange(recat, "B ... D -> B D ...")
        new_feats = dino.backbone(recat, pos)
        new_feats = rearrange(new_feats, "B D ... -> B ... D")
        if dino.new_target_key == "only_reg":
            latent = new_feats[:, :dino.nr_new_reg]
        elif dino.new_target_key == "only_nonreg":
            latent = new_feats[:, dino.nr_new_reg:]
            if not dino.use_reflatten:
                latent = rearrange(latent, "B (T N) D -> B T N D", T=nr_frames, N=N_og)
        elif dino.new_target_key == "all":
            latent = new_feats
        if self.decoder.use_pos_extra:
            B, T, N, D = latent.shape
            pos_extra = make_axial_pos_2d(1, 1, device=latent.device)
            pos_extra = repeat(pos_extra, " 1 D -> B N D", B=B, N=N)
            pos_extra = rearrange(pos_extra, "B ... D -> B (...) D")
            latent = rearrange(latent, "B ... D -> B (...) D")
        result = {
            # "cond_norm": cond_time,
            "x_extra": latent,
        }
        if self.decoder.use_pos_extra:
            result["pos_extra"] = pos_extra
        return result
    def sample(self, cls_token, z=None, sample_steps=50, dtype=torch.float32, device=torch.device("cuda"), seed=42):
        B = cls_token.shape[0]
        C, *DIMS = self.decoder.val_shape
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * B, device=z.device, dtype=z.dtype).view([B, *([1] * len(z.shape[1:]))])
        pos = self.decoder.get_pos(z)
        if z is None:
            z = torch.randn((B, C, *DIMS), dtype=dtype, generator=torch.manual_seed(seed)).to(
                    device
            )
        cond_dict = self.get_conditioning(cls_token=cls_token)
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * B, device=z.device, dtype=z.dtype)
            cond_time = self.get_time_cond(t)
            cond_dict["cond_norm"] = cond_time
            vc_cond = self.decoder.backbone(z, pos=pos, **cond_dict)
            z = z - dt * vc_cond
        recon = self.decoder.ae.decode(z)
        return recon
    @torch.no_grad()
    def decode(
        self, cls_token:torch.Tensor, **kwargs,
    ):
        """Decode/generate images from cls-token conditioning.

        Args:
            cls_token: Tensor of shape `(B, D)` produced by `RepTokImageEncoder`.
            **kwargs: Forwarded to `sample()` (e.g. `z`, `sample_steps`, `device`).
        """
        return self.sample(cls_token, **kwargs)
class RepTokImageEncoder(nn.Module):
    """DINO-based image encoder that returns cls-token conditioning features.

    Inputs are expected in `[-1, 1]` with shape `(B, C, H, W)`. Images are resized
    and normalized with DINO preprocessing before extracting the cls token `(B, D)`.
    """
    def __init__(
        self,
        base_url = "facebookresearch/dinov2",
        model_version = "dinov2_vitb14_reg",
        raw_ckpt = None,
        ckpt_path = None,
        normalize_mean: float | tuple[float, float, float] = [0.485, 0.456, 0.406],
        normalize_std: float | tuple[float, float, float] = [0.229, 0.224, 0.225],
        model_size: int = 224,
        device = torch.device("cpu"),
        dtype = torch.bfloat16,
    ):
        super().__init__()
        if raw_ckpt is not None:
            weights = raw_ckpt
        elif ckpt_path is not None:
            weights = torch.load(ckpt_path)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="xFormers is available.*")
                og_dino: torch.nn.Module = torch.hub.load(base_url, model_version, source="github")
            weights = og_dino.state_dict()
        self.encoder = vit_base_reg()
        self.encoder.load_state_dict(weights)
        self.encoder = self.encoder.to(device=device, dtype=dtype)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder.core_computation = torch.compile(
            self.encoder.core_computation, fullgraph=True, dynamic=False, mode="reduce-overhead"
        )
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.model_size = model_size
    def forward(
        self,
        imgs, # B C H W"
    ):
        """Encode images to DINO cls-token features.

        Args:
            imgs: Tensor `(B, C, H, W)` in `[-1, 1]`.

        Returns:
            Tensor `(B, D)` cls-token features.
        """
        assert imgs.min() >= -1.001, f"out of bounds {imgs.min()} {imgs.max()}"
        assert imgs.max() <= 1.001, f"out of bounds {imgs.min()} {imgs.max()}"
        assert len(imgs.shape) == 4
        imgs = better_resize(imgs, self.model_size)
        B, _, H, W = imgs.shape
        imgs = (imgs + 1.0) / 2.0  # to 0-1 range
        # copied from transformers preprocessor
        imgs = TVTF.normalize(imgs, self.normalize_mean, self.normalize_std)
        feats:dict[str, torch.Tensor] = self.encoder.forward_features(imgs.clone())
        cls_token = feats["x_norm_clstoken"]
        return cls_token


# Backward-compatible aliases for older configs and checkpoints.
RFAEEncoder = ConditionEncoderBase
ZipTokEncoder = RepTokEncoder
DinoEncoder = DINOFeatureEncoder
FramewiseDinoEncoder = MultiFrameDINOEncoder
VideoDinoEncoder = RepTokConditionEncoder
UnDINOLatentRF2D = RepTokLatentRF2D
UnDINO2DWrapper = RepTokDecoderWrapper
UnDINOEncoder = RepTokImageEncoder

# ---------------------------------------------------------------------
# Model loading helpers (note: only UnDINO without bridge is supported)
# ---------------------------------------------------------------------

def omegaconf_resolvers():
    def _resolver_exact_int_div(a, b):
        div = a // b
        assert div * b == a, f"exact_int_div resolver: {a} cannot be exactly divided by {b}."
        return div
    def _resolver_call(func, kwargs):
        return func(**kwargs)
    if OmegaConf._get_resolver("mul") is not None:
        return
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    OmegaConf.register_new_resolver("add", lambda a, b: a + b)
    OmegaConf.register_new_resolver("round", lambda v: round(v))
    OmegaConf.register_new_resolver("exact_int_div", _resolver_exact_int_div)
    OmegaConf.register_new_resolver("if", lambda a, b, c: b if a else c)
    OmegaConf.register_new_resolver("locate", lambda name: locate(name))
    OmegaConf.register_new_resolver("int", lambda s: int(s))
    OmegaConf.register_new_resolver("call", _resolver_call)
def recursive_fix_class_dscr(mapping:MutableMapping, prefix_to_replace:str="diffusion.model.", replacement_prefix:str="undino_monolith"):
    for key in mapping.keys():
        val = mapping[key]
        if isinstance(val, str):
            if val.startswith(prefix_to_replace):
                parts = val.split(".")
                val = f"{replacement_prefix}.{parts[-1]}"
        elif isinstance(val, MutableMapping):
            val = recursive_fix_class_dscr(val, prefix_to_replace=prefix_to_replace, replacement_prefix=replacement_prefix)
        mapping[key] = val
    return mapping
def load_raw(conf_path:str, ckpt_path:str=None, is_model_conf:bool=False, replacement_prefix=None) -> RepTokLatentRF2D:
    """Load a UnDINO monolith model from Hydra config, optionally with checkpoint.

    Args:
        conf_path: Hydra experiment config path (or model config if
            `is_model_conf=True`).
        ckpt_path: Optional checkpoint path containing the model `state_dict`.
            If omitted or `None`, the model is instantiated from config only.
        is_model_conf: If true, instantiate `conf_path` directly; otherwise use
            `conf.model`.
        replacement_prefix: Module prefix used to rewrite serialized class paths.
            If omitted, this file's module name is used and its directory is added
            to `sys.path`.

    Returns:
        Instantiated `RepTokLatentRF2D`. If `ckpt_path` is provided, weights are
        loaded on CPU.
    """
    assert os.path.isfile(conf_path), f"{conf_path=} not found"
    if ckpt_path is not None:
        assert os.path.isfile(ckpt_path), f"{ckpt_path=} not found"
    omegaconf_resolvers()
    conf = OmegaConf.load(conf_path)
    if is_model_conf:
        model_conf = conf
    else:
        model_conf = conf.model
    if replacement_prefix is None:
        this_file = __file__
        this_dir = os.path.dirname(this_file)
        this_basename = os.path.basename(this_file)
        this_filename = os.path.splitext(this_basename)[0]
        sys.path.append(this_dir)
        replacement_prefix = this_filename
    model_conf = recursive_fix_class_dscr(model_conf, replacement_prefix=replacement_prefix)
    model = hydra.utils.instantiate(model_conf)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model

# ---------------------------------------------------------------------
# Demonstration / CLI helper
# ---------------------------------------------------------------------

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, default=None, help="Path to an experiment yaml file.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a corresponding model checkpoint file.")
    parser.add_argument("--replacement_prefix", type=str, default=None, help="Python path to this module. If not set, the sys.path will be modified automatically.")
    parser.add_argument("--is_model_conf", action="store_true")
    return parser

def main(args:Namespace):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sample_steps = 50
    trained_model:RepTokLatentRF2D = load_raw(
        conf_path=args.conf, 
        ckpt_path=args.ckpt, 
        is_model_conf=args.is_model_conf,
        replacement_prefix=args.replacement_prefix,
    )
    trained_model = trained_model.to(device=device, dtype=dtype).eval()
    encoder = RepTokImageEncoder(raw_ckpt=trained_model.encoder.dino.model.state_dict(), device=device, dtype=dtype)
    decoder = RepTokDecoderWrapper(trained_model)
    x = torch.randn((16,3,256,256), device=device, dtype=dtype).clamp(-1,1)
    z = torch.randn((16,4,32,32), device=device, dtype=dtype)
    x_recon_v1 = trained_model.sample(z, sample_steps=sample_steps, x_cond=x)
    latent = encoder.forward(x)
    x_recon_v2 = decoder.decode(latent, z=z, sample_steps=sample_steps, device=device, dtype=dtype)
    assert torch.allclose(x_recon_v1, x_recon_v2)
    print(f"Sweet success!", flush=True)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
