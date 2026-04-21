import torch
import torch.nn as nn

from torchmetrics import SumMetric
from torchmetrics import MeanMetric
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


def calculate_PSNR(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    psnrs = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnrs.mean()


class ImageMetricTracker(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ssim_fn = SSIM(data_range=1.)
        self.ssim = MeanMetric()
        self.psnr = MeanMetric()
        self.mse = MeanMetric()
        self.total_samples = SumMetric()

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        )

    def __call__(self, target, pred):
        """ Assumes target and pred in [-1, 1] range """
        bs = target.shape[0]
        real_ims = un_normalize_ims(target)
        fake_ims = un_normalize_ims(pred)
        
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        # SSIM, PSNR, and MSE
        ssim_value = self.ssim_fn(pred / 2 + 0.5, target / 2 + 0.5)
        psnr_value = calculate_PSNR(pred / 2 + 0.5, target / 2 + 0.5)
        mse_value = nn.functional.mse_loss(pred, target)

        # Update metrics
        self.ssim.update(ssim_value)
        self.psnr.update(psnr_value)
        self.mse.update(mse_value)
        self.total_samples.update(bs)

    def reset(self):
        self.ssim.reset()
        self.psnr.reset()
        self.mse.reset()
        self.fid.reset()
        self.total_samples.reset()

    def aggregate(self):
        """ Compute the final metrics (automatically synced across devices) """
        return {
            "fid": self.fid.compute(),
            "ssim": self.ssim.compute(),
            "psnr": self.psnr.compute(),
            "mse": self.mse.compute(),
            "n_metric_samples": self.total_samples.compute(),
        }
