import torch
from pytorch_lightning import LightningModule
import numpy as np
import warnings

import os, sys
while os.path.basename(os.getcwd()) != "ziptok-clean":
    os.chdir("..")
sys.path.append(os.getcwd())

from reptok.encoder.encoder import RepTokImageEncoder, RepTokDecoderWrapper
from reptok.encoder.encoder import load_raw

class TrainerAutoencoder(LightningModule):
    def __init__(self, 
                 config_path,
                 ckpt_path=None,
                 image_size = 256,
                 scale_factor = 1.0,
                #  use_proj_token = True,
                 ):
        super().__init__()
        
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.num_channels = 4
        self.scale_factor = scale_factor
        # self.use_proj_token = use_proj_token

        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"TrainerAutoencoder requires a valid monolith config_path. Not found: {config_path!r}"
            )

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            trained_model = load_raw(config_path, ckpt_path)
        elif ckpt_path is None:
            trained_model = load_raw(config_path, ckpt_path=None)
        else:
            warnings.warn(
                "Monolith ckpt_path is missing/unavailable. Instantiating from config only and "
                "relying on the outer Lightning checkpoint to restore first-stage weights."
            )
            trained_model = load_raw(config_path, ckpt_path=None)
        self.encoder = RepTokImageEncoder(raw_ckpt = trained_model.encoder.dino.model.state_dict())
        self.decoder = RepTokDecoderWrapper(trained_model)

    @torch.no_grad()
    @torch.amp.autocast("cuda")
    def encode(self, x):
        latent = self.encoder.forward(x)
        latent = latent.unsqueeze(1)
        latent = latent / self.scale_factor
        return latent

    @torch.no_grad()
    @torch.amp.autocast("cuda")
    def decode(self, x):
        x = x * self.scale_factor
        x = x.squeeze(1)

        noise = torch.randn(x.shape[0], self.num_channels, self.latent_size, self.latent_size).to(x.device)
        out = self.decoder.decode(x, z=noise, sample_steps=50, device=x.device)

        return out
