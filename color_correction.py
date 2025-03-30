import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ColorCorrection(nn.Module):
    def __init__(
        self,
        levels: int = 5,
        kernel_size: int = 3,
    ):
        """
        NCHW tensors with N=1 and C=3
        RGB orderd channels
        """
        super().__init__()
        sigma: float = 1.0
        x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gaussian = torch.exp(-x**2 / (2 * sigma**2))
        gaussian /= gaussian.sum()
        self.kernel: Tensor = (
            gaussian
            .outer(gaussian)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(3, 1, 1, 1)
        )

        self.levels: int = levels


    @staticmethod
    def multi_scale_blur(x: torch.Tensor, dilation: int, kernel: Tensor) -> Tensor:
        pad = dilation * ((kernel.shape[2] - 1) // 2)
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(x, kernel, groups=3, dilation=dilation)


    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        h, w = x.shape[2:]
        ref: Tensor = F.interpolate(
            input=ref,
            size=(h, w),
            mode='bicubic',
            align_corners=True,
            antialias=True
        )

        kernel: Tensor = self.kernel.to(device=x.device, dtype=x.dtype)

        ref_low_freq = ref
        x_high_freq = torch.zeros_like(x)
        for i in range(self.levels):
            dilation = 2**i

            x_low_freq = self.multi_scale_blur(x, dilation=dilation, kernel=kernel)
            x_high_freq += x - x_low_freq
            x = x_low_freq

            ref_low_freq = self.multi_scale_blur(ref_low_freq, dilation=dilation, kernel=kernel)

        return x_high_freq + ref_low_freq

