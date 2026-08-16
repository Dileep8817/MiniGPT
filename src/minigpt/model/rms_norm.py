
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    # llama-style: no mean subtraction, no bias, just x / rms(x) * weight
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # rms in fp32 so bf16 stays stable, then cast back before the affine scale
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight
