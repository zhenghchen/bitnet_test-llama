import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from bitnet import BitLinear # Import your custom layer

def convert_to_bitnet(module, copy_weights=False, module_name=""):
    """
    Recursively convert LLaMA model's nn.Linear layers within Attention and MLP blocks
    to BitLinear layers for Quantization-Aware Pre-training.

    Args:
        module: The PyTorch module (or model) to convert.
        copy_weights (bool): Whether to copy weights. Should be False for pre-training.
        module_name (str): Internal tracking of module names for debugging.
    """
    for name, child_module in module.named_children():
        full_child_name = f"{module_name}.{name}" if module_name else name
        # Target specific Linear layers within LlamaAttention and LlamaMLP
        if isinstance(child_module, nn.Linear) and \
           any(isinstance(module, t) for t in [LlamaAttention, LlamaMLP]):

            print(f"Replacing {full_child_name} with BitLinear")
            bitlinear_layer = BitLinear(
                child_module.in_features,
                child_module.out_features,
                bias=child_module.bias is not None
            ).to(device=child_module.weight.device, dtype=child_module.weight.dtype) # Match device and dtype

            if copy_weights:
                # This block should not run during pre-training from scratch
                print(f"WARNING: Copying weights for {full_child_name} - ensure this is intended (e.g., for fine-tuning, not pre-training)")
                with torch.no_grad():
                    bitlinear_layer.weight.copy_(child_module.weight)
                    if child_module.bias is not None:
                         if bitlinear_layer.bias is not None:
                            bitlinear_layer.bias.copy_(child_module.bias)
                         else:
                            print(f"Warning: Original layer {full_child_name} had bias, but new BitLinear layer does not.")


            # Replace the original layer with the new BitLinear layer
            setattr(module, name, bitlinear_layer)

        # Keep original RMSNorm layers - do not replace them with Identity
        # Recursively apply to children
        else:
            convert_to_bitnet(child_module, copy_weights=copy_weights, module_name=full_child_name)