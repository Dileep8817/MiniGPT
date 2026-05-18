
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm (Llama-style).
    Like LayerNorm but:
      * no mean subtraction (only RMS scaling)
      * no bias parameter
      * normalization is computed in fp32 for numerical stability under
        bf16, then cast back to the input dtype before the affine scale.
    Output: x / RMS(x) * weight, where RMS(x) = sqrt(mean(x^2) + eps).
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight
    
