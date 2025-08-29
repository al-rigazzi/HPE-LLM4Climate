"""
AIFS Model Structure Summary

This script provides comprehensive PyTorch module structure analysis for the AIFS model,
including layers, parameters, and architectural details.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch.nn as nn

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from aifs_wrapper import AIFSWrapper


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_layer_info(
    module: nn.Module, name: str = "", max_depth: int = 3, current_depth: int = 0
) -> List[Dict[str, Any]]:
    """Recursively extract layer information from a PyTorch module."""
    layers = []

    if current_depth >= max_depth:
        return layers

    for child_name, child_module in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        # Get parameter count for this specific layer
        layer_params = sum(p.numel() for p in child_module.parameters(recurse=False))
        total_layer_params = sum(p.numel() for p in child_module.parameters())

        # Get layer type and shape information
        layer_info = {
            "name": full_name,
            "type": type(child_module).__name__,
            "layer_params": layer_params,
            "total_params": total_layer_params,
            "trainable": any(p.requires_grad for p in child_module.parameters()),
        }

        # Add specific information for common layer types
        if hasattr(child_module, "weight") and child_module.weight is not None:
            layer_info["weight_shape"] = list(child_module.weight.shape)

        if hasattr(child_module, "bias") and child_module.bias is not None:
            layer_info["bias_shape"] = list(child_module.bias.shape)

        if hasattr(child_module, "in_features"):
            layer_info["in_features"] = child_module.in_features

        if hasattr(child_module, "out_features"):
            layer_info["out_features"] = child_module.out_features

        if hasattr(child_module, "num_heads"):
            layer_info["num_heads"] = child_module.num_heads

        if hasattr(child_module, "embed_dim"):
            layer_info["embed_dim"] = child_module.embed_dim

        layers.append(layer_info)

        # Recursively get child layers
        child_layers = get_layer_info(child_module, full_name, max_depth, current_depth + 1)
        layers.extend(child_layers)

    return layers


def analyze_attention_layers(model: nn.Module) -> List[Dict[str, Any]]:
    """Find and analyze attention layers in the model."""
    attention_layers = []

    def find_attention(module: nn.Module, name: str = ""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check for various attention layer types
            if any(
                attn_type in type(child_module).__name__.lower()
                for attn_type in ["attention", "multihead", "selfattn", "crossattn"]
            ):

                attn_info = {
                    "name": full_name,
                    "type": type(child_module).__name__,
                    "parameters": sum(p.numel() for p in child_module.parameters()),
                }

                # Extract attention-specific attributes
                for attr in ["num_heads", "embed_dim", "head_dim", "dropout"]:
                    if hasattr(child_module, attr):
                        attn_info[attr] = getattr(child_module, attr)

                attention_layers.append(attn_info)

            # Recurse into children
            find_attention(child_module, full_name)

    find_attention(model)
    return attention_layers


def print_model_summary(model: nn.Module):
    """Print comprehensive model summary."""
    print("=" * 80)
    print("🤖 AIFS PyTorch Model Structure Summary")
    print("=" * 80)

    # Basic model information
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"   Model Type: {type(model).__name__}")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"   Model Size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32

    # Model architecture overview
    print(f"\n🏗️ Model Architecture:")
    print(f"   Module: {model.__class__.__module__}")
    print(f"   Class: {model.__class__.__name__}")

    # High-level module breakdown
    print(f"\n📋 Top-Level Modules:")
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        percentage = (child_params / total_params) * 100 if total_params > 0 else 0
        print(
            f"   {name:30} {type(child).__name__:25} {child_params:>12,} params ({percentage:5.1f}%)"
        )

    # Detailed layer analysis
    print(f"\n🔍 Detailed Layer Analysis (Top 3 levels):")
    layers = get_layer_info(model, max_depth=3)

    print(f"{'Layer Name':40} {'Type':20} {'Parameters':15} {'Details':30}")
    print("-" * 105)

    for layer in layers[:50]:  # Limit to first 50 layers to avoid overwhelming output
        details = []
        if "weight_shape" in layer:
            details.append(f"W:{layer['weight_shape']}")
        if "in_features" in layer and "out_features" in layer:
            details.append(f"{layer['in_features']}→{layer['out_features']}")
        if "num_heads" in layer:
            details.append(f"heads:{layer['num_heads']}")

        detail_str = ", ".join(details[:3])  # Limit detail string length

        print(
            f"{layer['name'][:39]:40} {layer['type'][:19]:20} {layer['total_params']:>12,} {detail_str[:29]:30}"
        )

    if len(layers) > 50:
        print(f"   ... and {len(layers) - 50} more layers")

    # Attention layer analysis
    print(f"\n🎯 Attention Layer Analysis:")
    attention_layers = analyze_attention_layers(model)

    if attention_layers:
        print(f"   Found {len(attention_layers)} attention layers:")
        for attn in attention_layers:
            print(f"   {attn['name']:40} {attn['type']:20} {attn['parameters']:>12,} params")
            if "num_heads" in attn:
                print(
                    f"      ↳ Heads: {attn['num_heads']}, Embed Dim: {attn.get('embed_dim', 'N/A')}"
                )
    else:
        print("   No explicit attention layers found (may use alternative attention mechanisms)")

    # Model device and dtype information
    print(f"\n💾 Model Properties:")
    try:
        first_param = next(model.parameters())
        print(f"   Device: {first_param.device}")
        print(f"   Data Type: {first_param.dtype}")
        print(f"   Requires Grad: {first_param.requires_grad}")
    except StopIteration:
        print("   No parameters found")

    # Training mode
    print(f"   Training Mode: {model.training}")

    print("\n" + "=" * 80)


def main():
    """Main function to analyze AIFS model structure."""
    try:
        print("🚀 Initializing AIFS model for structure analysis...")

        # Initialize AIFS wrapper
        aifs = AIFSWrapper()

        # Load the model
        model_info = aifs._load_model()

        if model_info["pytorch_model"] is None:
            print("❌ Could not access PyTorch model for analysis")
            print("   The model is only available through SimpleRunner interface")
            return

        pytorch_model = model_info["pytorch_model"]
        print(f"✅ Successfully loaded PyTorch model: {type(pytorch_model).__name__}")

        # Perform comprehensive analysis
        print_model_summary(pytorch_model)

        # Additional model-specific information
        print(f"\n🌍 AIFS-Specific Information:")
        model_info_dict = aifs.get_model_info()
        for key, value in model_info_dict.items():
            print(f"   {key}: {value}")

        print(f"\n📝 Available Variables ({len(aifs.get_available_variables())}):")
        variables = aifs.get_available_variables()
        for i, var in enumerate(variables):
            if i % 3 == 0:
                print(f"   ", end="")
            print(f"{var:30}", end="")
            if (i + 1) % 3 == 0:
                print()
        if len(variables) % 3 != 0:
            print()

    except Exception as e:
        print(f"❌ Error during model analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
