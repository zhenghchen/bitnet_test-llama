import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM

# Define quantization functions
def activation_quant(x):
    #Quantize activations to 8 bits.
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

# Define the BitLinear class
class BitLinear(nn.Linear):
    #Replace nn.Linear with BitLinear.
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        # Quantize weights
        w = self.weight
        w_quant = w + (weight_quant(w) - w).detach()

        # Quantize activations
        x_quant = x + (activation_quant(x) - x).detach()

        # Perform linear transformation
        y = F.linear(x_quant, w_quant, self.bias)
        return y

# Modify the LLaMA model
class BitLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.replace_linears()

    def replace_linears(self, module=None):
        #Replace all nn.Linear layers with BitLinear.
        if module is None:
            module = self

        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace nn.Linear with BitLinear
                bit_linear = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
                bit_linear.weight = child.weight
                bit_linear.bias = child.bias
                setattr(module, name, bit_linear)
            elif hasattr(child, 'children'):
                self.replace_linears(child)

# help with debug
print("Loading LLaMA 2 model...")
model = BitLlama.from_pretrained("models/Llama-2-7b-chat-hf")

print("Saving quantized model...")
model.save_pretrained("models/Llama-2-7b-chat-hf-bitnet")
print("it work")