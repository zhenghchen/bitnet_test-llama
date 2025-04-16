import torch
import torch.nn as nn
import torch.nn.functional as F

# Removed LlamaRMSNorm import

def activation_quant(x):
    """Per-token quantization to 8 bits using absmax. STE is implicit."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    # Quantize and dequantize using the Straight-Through Estimator (STE) trick
    # The gradient will flow back through the original 'x' values.
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    """Per-tensor quantization to 1.58 bits (ternary values). STE is implicit."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and dequantize using the Straight-Through Estimator (STE) trick
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    """
    BitLinear layer for Quantization-Aware Training (QAT).
    Assumes input x is already normalized by a preceding layer (e.g., RMSNorm).
    Uses Straight-Through Estimator (STE) for gradients.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        # Optional: Ensure input is on the correct device (often handled by Trainer/Accelerator)
        # x = x.to(self.weight.device)

        # Input 'x' is assumed to be normalized by a layer before this one.

        # STE for activations: Apply quantization function, gradient flows through original 'x'
        # The STE logic is embedded in the quantization functions now.
        x_quant = x + (activation_quant(x) - x).detach()

        # STE for weights: Apply quantization function, gradient flows through original 'self.weight'
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # Linear operation using STE-quantized values but standard precision math
        output = F.linear(x_quant, w_quant, self.bias)
        return output