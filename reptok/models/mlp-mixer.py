# Sourced from https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

from reptok.models.transformer.dit import TimestepEmbedder, LabelEmbedder

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

def MLPMixer_1D(n_channels, channels, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        nn.Linear(channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(n_channels, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Linear(dim, channels)
    )

class MLPMixer1D(nn.Module):
    def __init__(self, n_channels, channels, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0., num_classes = 0, class_dropout_prob = 0.1, n_extra_tokens = 0, zero_init_last = False):
        super().__init__()
        self.n_channels = n_channels
        self.channels = channels
        mlp_ch = n_channels + 2 if num_classes > 0 else n_channels + 1
        mlp_ch += n_extra_tokens
        self.mixer = MLPMixer_1D(mlp_ch, channels, dim, depth, expansion_factor, expansion_factor_token, dropout)
        if zero_init_last:
            torch.nn.init.zeros_(self.mixer[-1].weight)
            torch.nn.init.zeros_(self.mixer[-1].bias)

        self.mixer = torch.compile(self.mixer)

        self.pos_embed = nn.Parameter(torch.randn(1, n_channels, channels))
        self.t_embedder = TimestepEmbedder(channels)
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        if num_classes > 0:
            self.label_embedder = LabelEmbedder(num_classes, channels, class_dropout_prob)
        else:
            self.label_embedder = None

    def forward(self, x, t, concat_tokens=None, y=None):
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (b, n_channels, channels).
        """
        n_ch = x.shape[1]
        x = x + self.pos_embed
        t = self.t_embedder(t).unsqueeze(1)
        y = self.label_embedder(y, self.training).unsqueeze(1) if self.label_embedder is not None else None

        xs = torch.concat([x, t], dim=1) if y is None else torch.cat([x, y, t], dim=1)
        if concat_tokens is not None:
            n_extra_tokens = concat_tokens.shape[1]
            xs = torch.cat([concat_tokens, xs], dim=1)
        else:
            n_extra_tokens = 0

        # Apply the mixer
        x = self.mixer(xs)

        # Remove context tokens
        x = x[:, n_extra_tokens:]
        # Extract the output
        x = x[:, :n_ch]

        return x
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        raise NotImplementedError("Not implemented yet.")
    

if __name__ == "__main__":
    model = MLPMixer1D(
        n_channels=1,
        channels=1152,
        dim=1536,
        depth=28,
        expansion_factor_token=4,
        expansion_factor=3,
        num_classes=1000,
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    x = torch.randn(1, 1, 1152)
    t = torch.rand(1)
    y = torch.randint(0, 1000, (1,))

    out = model(x, t, y=y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
