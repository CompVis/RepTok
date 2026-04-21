import wandb
import torch
import einops
from PIL import Image
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn

from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger

from jutils import instantiate_from_config
from jutils import load_partial_from_config
from jutils import exists, freeze, default

from reptok.metrics import ImageMetricTracker


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


class TrainerModuleLatentFlow(LightningModule):
    def __init__(
        self,
        flow_cfg: dict,
        tokenizer_cfg: dict,
        first_stage_cfg: dict = None,
        lr: float = 1e-4,
        weight_decay: float = 0.,
        ema_rate: float = 0.99,
        lr_scheduler_cfg: dict = None,
        # logging
        log_grad_norm: bool = False,
        n_images_to_vis: int = 16,
        sample_kwargs: dict = None
    ):
        super().__init__()

        self.model = instantiate_from_config(flow_cfg)
        
        # EMA model
        self.ema_model = None
        self.ema_rate = ema_rate
        if ema_rate > 0:
            self.ema_model = deepcopy(self.model)
            freeze(self.ema_model)
            self.ema_model.eval()
            update_ema(self.ema_model, self.model, decay=0)     # ensure EMA is in sync
        
        # first stage
        self.first_stage = None
        if exists(first_stage_cfg):
            self.first_stage = instantiate_from_config(first_stage_cfg)
            freeze(self.first_stage)
            self.first_stage.eval().to(self.device)

        # encoder/tokenizer
        if exists(tokenizer_cfg):
            self.tokenizer = instantiate_from_config(tokenizer_cfg).to(self.device)
        else:
            self.tokenizer = None

        # training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        # visualization
        self.sample_kwargs = sample_kwargs or {}
        self.n_images_to_vis = n_images_to_vis
        self.image_shape = None
        self.latent_shape = None
        self.generator = torch.Generator()

        # evaluation
        self.metric_tracker = ImageMetricTracker()

        # SD3 & Meta Movie Gen show that val loss correlates with human quality
        # and compute the loss in equidistant segments in (0, 1) to reduce variance
        self.val_losses = []        # only for Flow model
        self.val_images = None
        self.val_epochs = 0

        self.save_hyperparameters()

        # signal handler for slurm, flag to make sure the signal
        # is not handled at an incorrect state, e.g. during weights update
        self.stop_training = False

    # dummy function to be compatible
    def stop_training_method(self):
        pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay
        )
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out
    
    @torch.no_grad()
    def encode(self, x):
        return self.first_stage.encode(x) if exists(self.first_stage) else x
    
    @torch.no_grad()
    def decode(self, z):
        return self.first_stage.decode(z) if exists(self.first_stage) else z
    
    def training_step(self, batch, batch_idx):
        ims = batch["image"]
        latent = batch.get("latent", self.encode(ims))

        # encode the images into tokens
        tokens = self.tokenizer(ims)            # (b, n, d)
        tokens = tokens['proj_token']

        # compute loss
        loss = self.model.training_losses(latent, concat_tokens=tokens)

        # logging, ema, scheduler, etc
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=ims.shape[0], sync_dist=False)

        if exists(self.ema_model): update_ema(self.ema_model, self.model, decay=self.ema_rate)
        if exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if self.stop_training: self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        ims = batch["image"]
        latent = default(batch.get("latent"), None)
        tokens = self.tokenizer(ims)
        tokens = tokens['proj_token']
        bs = ims.shape[0]
        
        # only flow models val loss shows correlation with human quality
        if hasattr(self.model, 'validation_losses'):
            latent = default(latent, self.encode(ims))
            _, val_loss_per_segment = self.model.validation_losses(latent, concat_tokens=tokens)
            self.val_losses.append(val_loss_per_segment)

        # generation
        if self.latent_shape is None:
            _latent = self.encode(ims)
            self.latent_shape = _latent.shape[1:]
            self.batch_size = bs
        
        # sample images
        g = self.generator.manual_seed(batch_idx)
        z = torch.randn((bs, *self.latent_shape), generator=g).to(self.device)
        sampler = self.ema_model if exists(self.ema_model) else self.model
        samples = sampler.generate(x=z, concat_tokens=tokens, **self.sample_kwargs)
        samples = self.decode(samples)
        
        # metrics
        self.metric_tracker(ims, samples)

        # save the images for visualization
        if self.val_images is None:
            real_ims = un_normalize_ims(ims)
            fake_ims = un_normalize_ims(samples)
            self.val_images = {
                "real": real_ims[:self.n_images_to_vis],
                "fake": fake_ims[:self.n_images_to_vis],
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
