import wandb
import torch
import einops
import warnings
from PIL import Image
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn

from lightning.pytorch.loggers import WandbLogger
from jutils import exists, default

import os, sys
while os.path.basename(os.getcwd()) != "ziptok-clean":
    os.chdir("..")
sys.path.append(os.getcwd())

from reptok.trainer import TrainerModuleLatentFlow as BaseTrainerModuleLatentFlow


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


def ims_to_grid(ims, stack="row", split=4):
    """ Convert (b, c, h, w) to (h, w, c) """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    if split is not None and ims.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        ims = einops.rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        ims = einops.rearrange(ims, "b c h w -> " + to)
    return ims


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        # unwrap DDP
        if name.startswith('module.'):
            name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class TrainerModuleLatentFlow(BaseTrainerModuleLatentFlow):
    def __init__(
        self,
        flow_cfg: dict,
        tokenizer_cfg: dict,
        first_stage_cfg: dict = None,
        lr: float = 1e-4,
        weight_decay: float = 0.,
        ema_rate: float = 0.99,
        lr_scheduler_cfg: dict = None,
        log_grad_norm: bool = False,
        n_images_to_vis: int = 16,
        sample_kwargs: dict = None,
        # cond args
        cond_stage_key: str = "label",
        cond_stage_method: str = "y",
        cond_dropout_prob: float = 0.,
    ):
        super().__init__(
            flow_cfg=flow_cfg,
            tokenizer_cfg=tokenizer_cfg,
            first_stage_cfg=first_stage_cfg,
            lr=lr,
            weight_decay=weight_decay,
            ema_rate=ema_rate,
            lr_scheduler_cfg=lr_scheduler_cfg,
            log_grad_norm=log_grad_norm,
            n_images_to_vis=n_images_to_vis,
            sample_kwargs=sample_kwargs
        )

        self.cond_stage_key = cond_stage_key
        self.cond_stage_method = cond_stage_method
        self.cond_dropout_prob = cond_dropout_prob

    @torch.no_grad()
    def generate_with_cfg_interval(self, x, **kwargs):
        sampler = self.ema_model if exists(self.ema_model) else self.model
        return sampler.generate_with_cfg_interval(x=x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        ims = batch["image"]
        latent = batch.get("latent", self.encode(ims))

        conditioning = batch.get(self.cond_stage_key, None)
        if exists(conditioning) and self.cond_dropout_prob > 0.:
            assert exists(self.tokenizer), "Conditioning dropout in the trainer module requires a tokenizer"
            uncond = self.tokenizer.get_unconditional_embedding()
            dropout_mask = torch.rand(conditioning.shape[0], device=conditioning.device) < self.cond_dropout_prob
            conditioning[dropout_mask] = uncond

        # compute loss
        conditioning_dict = {self.cond_stage_method: conditioning}
        loss = self.model.training_losses(latent, **conditioning_dict)

        # logging, ema, scheduler, etc
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=ims.shape[0], sync_dist=True)

        if exists(self.ema_model): update_ema(self.ema_model, self.model, decay=self.ema_rate)
        if exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if self.stop_training: self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        ims = batch["image"]
        latent = default(batch.get("latent"), None)
        bs = ims.shape[0]

        conditioning = self._prepare_conditioning(batch.get(self.cond_stage_key, None))
        
        conditioning_dict = {self.cond_stage_method: conditioning}
        # only flow models val loss shows correlation with human quality
        if hasattr(self.model, 'validation_losses'):
            latent = default(latent, self.encode(ims))
            _, val_loss_per_segment = self.model.validation_losses(latent, **conditioning_dict)
            self.val_losses.append(val_loss_per_segment)

        # generation
        if self.latent_shape is None:
            _latent = self.encode(ims)
            self.latent_shape = _latent.shape[1:]
            self.batch_size = bs

        latent = self.encode(ims)
        first_stage_reconstruction = self.decode(latent)
        
        # sample images
        g = self.generator.manual_seed(batch_idx)
        z = torch.randn((bs, *self.latent_shape), generator=g).to(self.device)
        sampler = self.ema_model if exists(self.ema_model) else self.model
        samples = sampler.generate(x=z, **conditioning_dict, **self.sample_kwargs)
        samples = self.decode(samples)
        
        # metrics
        self.metric_tracker(ims, samples)

        # save the images for visualization
        if self.val_images is None:
            real_ims = un_normalize_ims(ims)
            fake_ims = un_normalize_ims(samples)
            # first stage reconstructions
            first_stage_reconstructions = un_normalize_ims(first_stage_reconstruction)
            self.val_images = {
                "real": real_ims[:self.n_images_to_vis],
                "fake": fake_ims[:self.n_images_to_vis],
                "recon": first_stage_reconstructions[:self.n_images_to_vis],
            }

    def on_validation_epoch_end(self):
        # visualization
        for key, ims in self.val_images.items():
            self.log_images(ims, f"val/{key}", stack="row", split=4)
        
        # reset val images
        self.val_images = None

        # compute metrics
        metrics = self.metric_tracker.aggregate()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, sync_dist=True)
        self.metric_tracker.reset()

        # compute val loss if available (Flow models)
        if len(self.val_losses) > 0:
            val_losses = torch.stack(self.val_losses, 0)        # (N batches, segments)
            val_losses = val_losses.mean(0)                     # mean per segment
            # Don't log individual segment losses
            # for i, loss in enumerate(val_losses):
            #     self.log(f"val/loss_segment_{i}", loss, sync_dist=True)
            self.log("val/loss", val_losses.mean(), sync_dist=True)
            self.val_losses = []

        # log some information
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs:,} | Optimizer step {self.global_step:,}: {metrics['fid']:.2f} FID @ {int(metrics['n_metric_samples']):,} samples")

    def log_images(self, ims, name, stack="row", split=4):
        """
        Args:
            ims: torch.Tensor or np.ndarray of shape (b, c, h, w) in range [0, 255]
            name: str
        """
        ims = ims_to_grid(ims, stack=stack, split=split)
        if isinstance(ims, torch.Tensor):
            ims = ims.cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            ims = Image.fromarray(ims)
            ims = wandb.Image(ims)
            self.logger.experiment.log({f"{name}/samples": ims})
        else:
            ims = einops.rearrange(ims, "h w c -> c h w")
            self.logger.experiment.add_image(f"{name}/samples", ims, global_step=self.global_step)


def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

if __name__ == "__main__":
    import numpy as np
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`.*",
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module = TrainerModuleLatentFlow.load_from_checkpoint("checkpoints/reptok-xl-600k-ImageNet.ckpt",
                                                          map_location=dev,
                                                          strict=True,
                                                          )

    bs = 16
    y = torch.randint(0, 1000, (bs,)).to(dev)
    uc_cond = torch.tensor([1000]).to(dev).repeat(y.shape[0])

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        image_path = "assets/lion.jpg"
        image = Image.open(image_path).convert("RGB")
        image = image.resize((256, 256))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(dev) * 2 - 1  # scale to [-1, 1]

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            z = module.encode(image_tensor)
            decoded_samples = module.decode(z)

        decoded_image = decoded_samples[0].float().cpu().permute(1, 2, 0).numpy()
        decoded_image = (decoded_image + 1) / 2  # scale back to [0, 1]

        # calculate PSNR
        mse = np.mean((decoded_image - (image_tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2) ** 2)
        psnr = 10 * np.log10(1 / mse)
        print(f"PSNR: {psnr:.2f} dB")

    
