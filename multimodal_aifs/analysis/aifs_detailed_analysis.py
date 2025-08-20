"""
AIFS Detailed Architecture Analysis

Deep dive into the AIFS transformer architecture with focus on specific components.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from aifs_wrapper import AIFSWrapper


def analyze_transformer_blocks(model: nn.Module):
    """Analyze transformer block structure in detail."""
    print("\n🔧 Transformer Block Detailed Analysis:")
    print("=" * 60)

    # Find processor blocks
    processor = None
    for name, module in model.named_modules():
        if "processor" in name and hasattr(module, "proc"):
            processor = module
            break

    if processor is None:
        print("❌ Could not find processor module")
        return

    print(f"Processor type: {type(processor).__name__}")

    # Analyze the processor layers
    if hasattr(processor, "proc") and hasattr(processor.proc, "__iter__"):
        for layer_idx, layer in enumerate(processor.proc):
            print(f"\n📦 Processor Layer {layer_idx}:")
            print(f"   Type: {type(layer).__name__}")

            # Count parameters in this layer
            layer_params = sum(p.numel() for p in layer.parameters())
            print(f"   Parameters: {layer_params:,}")

            # Look for blocks within the layer
            if hasattr(layer, "blocks"):
                print(f"   Transformer Blocks: {len(layer.blocks)}")

                for block_idx, block in enumerate(layer.blocks):
                    print(f"\n   🧱 Block {block_idx}:")
                    print(f"      Type: {type(block).__name__}")

                    block_params = sum(p.numel() for p in block.parameters())
                    print(f"      Parameters: {block_params:,}")

                    # Analyze components within each block
                    for comp_name, component in block.named_children():
                        comp_params = sum(p.numel() for p in component.parameters())
                        print(
                            f"      {comp_name:15} {type(component).__name__:25} {comp_params:>10,} params"
                        )

                        # Special handling for attention layers
                        if "attention" in comp_name.lower():
                            if hasattr(component, "num_heads"):
                                print(f"         ↳ Heads: {component.num_heads}")
                            if hasattr(component, "embed_dim"):
                                print(f"         ↳ Embed Dim: {component.embed_dim}")
                            if hasattr(component, "head_dim"):
                                print(f"         ↳ Head Dim: {component.head_dim}")


def analyze_graph_structure(model: nn.Module):
    """Analyze graph neural network components."""
    print("\n🕸️ Graph Neural Network Analysis:")
    print("=" * 60)

    # Find encoder and decoder
    encoder, decoder = None, None
    for name, module in model.named_modules():
        if name.endswith("encoder") and "GraphTransformer" in type(module).__name__:
            encoder = module
        elif name.endswith("decoder") and "GraphTransformer" in type(module).__name__:
            decoder = module

    for component_name, component in [("Encoder", encoder), ("Decoder", decoder)]:
        if component is None:
            continue

        print(f"\n📡 {component_name}:")
        print(f"   Type: {type(component).__name__}")

        total_params = sum(p.numel() for p in component.parameters())
        print(f"   Total Parameters: {total_params:,}")

        # Analyze subcomponents
        for name, submodule in component.named_children():
            sub_params = sum(p.numel() for p in submodule.parameters())
            print(f"   {name:20} {type(submodule).__name__:25} {sub_params:>12,} params")

            # Look for embedding layers
            if "emb" in name:
                if hasattr(submodule, "weight"):
                    weight_shape = list(submodule.weight.shape)
                    print(f"      ↳ Weight shape: {weight_shape}")
                if hasattr(submodule, "in_features") and hasattr(submodule, "out_features"):
                    print(f"      ↳ {submodule.in_features} → {submodule.out_features} features")


def analyze_embedding_layers(model: nn.Module):
    """Analyze embedding and projection layers."""
    print("\n🎯 Embedding & Projection Analysis:")
    print("=" * 60)

    embeddings = []
    projections = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_info = {
                "name": name,
                "type": type(module).__name__,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "parameters": sum(p.numel() for p in module.parameters()),
                "weight_shape": list(module.weight.shape) if hasattr(module, "weight") else None,
                "has_bias": module.bias is not None,
            }

            if "emb" in name.lower():
                embeddings.append(module_info)
            else:
                projections.append(module_info)

    print(f"\n📌 Embedding Layers ({len(embeddings)}):")
    for emb in embeddings:
        print(
            f"   {emb['name']:40} {emb['in_features']:>6} → {emb['out_features']:>6} ({emb['parameters']:>10,} params)"
        )
        if emb["weight_shape"]:
            print(f"      ↳ Weight: {emb['weight_shape']}, Bias: {emb['has_bias']}")

    print(f"\n🎯 Key Linear Projections (Top 10 by parameters):")
    projections.sort(key=lambda x: x["parameters"], reverse=True)
    for proj in projections[:10]:
        print(
            f"   {proj['name']:40} {proj['in_features']:>6} → {proj['out_features']:>6} ({proj['parameters']:>10,} params)"
        )


def analyze_data_flow(model: nn.Module):
    """Analyze the data flow through the model."""
    print("\n🌊 Data Flow Analysis:")
    print("=" * 60)

    # Look for trainable tensors and their purposes
    trainable_tensors = []
    for name, module in model.named_modules():
        if "trainable" in name.lower() or "TrainableTensor" in type(module).__name__:
            if hasattr(module, "tensor"):
                tensor_shape = list(module.tensor.shape)
                tensor_params = module.tensor.numel()
                trainable_tensors.append(
                    {"name": name, "shape": tensor_shape, "parameters": tensor_params}
                )

    print(f"🔢 Trainable Tensors ({len(trainable_tensors)}):")
    for tensor_info in trainable_tensors:
        print(
            f"   {tensor_info['name']:35} Shape: {str(tensor_info['shape']):25} ({tensor_info['parameters']:>10,} params)"
        )

    # Analyze preprocessors and postprocessors
    print(f"\n⚙️ Data Processing Pipeline:")
    for proc_type in ["pre_processors", "post_processors"]:
        proc_module = None
        for name, module in model.named_modules():
            if name == proc_type:
                proc_module = module
                break

        if proc_module:
            print(f"   {proc_type:15}: {type(proc_module).__name__}")
            if hasattr(proc_module, "processors"):
                for proc_name, processor in proc_module.processors.items():
                    print(f"      ↳ {proc_name}: {type(processor).__name__}")


def analyze_model_capabilities(model: nn.Module):
    """Analyze model capabilities and constraints."""
    print("\n🎪 Model Capabilities Analysis:")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())

    # Estimate memory requirements
    param_memory_mb = total_params * 4 / (1024**2)  # float32
    activation_memory_estimate = param_memory_mb * 0.5  # Rough estimate
    total_memory_estimate = param_memory_mb + activation_memory_estimate

    print(f"💾 Memory Requirements:")
    print(f"   Parameters: {param_memory_mb:.1f} MB")
    print(f"   Estimated Activations: {activation_memory_estimate:.1f} MB")
    print(f"   Total Estimated: {total_memory_estimate:.1f} MB")

    # Analyze model architecture characteristics
    attention_layers = 0
    linear_layers = 0
    norm_layers = 0

    for module in model.modules():
        if "attention" in type(module).__name__.lower():
            attention_layers += 1
        elif isinstance(module, nn.Linear):
            linear_layers += 1
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            norm_layers += 1

    print(f"\n🏗️ Architecture Composition:")
    print(f"   Attention Layers: {attention_layers}")
    print(f"   Linear Layers: {linear_layers}")
    print(f"   Normalization Layers: {norm_layers}")

    # Model complexity metrics
    print(f"\n📈 Complexity Metrics:")
    print(f"   Parameter Density: {total_params / (1024**2):.2f} M params")
    if attention_layers > 0:
        print(f"   Params per Attention Layer: {total_params // attention_layers:,}")

    # Inference characteristics
    print(f"\n⚡ Inference Characteristics:")
    print(f"   Current Device: {next(model.parameters()).device}")
    print(f"   Data Type: {next(model.parameters()).dtype}")
    print(f"   Training Mode: {model.training}")


def main():
    """Main function for detailed architecture analysis."""
    try:
        print("🔬 AIFS Detailed Architecture Analysis")
        print("=" * 80)

        # Initialize and load model
        aifs = AIFSWrapper()
        model_info = aifs._load_model()

        if model_info["pytorch_model"] is None:
            print("❌ Could not access PyTorch model")
            return

        model = model_info["pytorch_model"]

        # Run all analyses
        analyze_transformer_blocks(model)
        analyze_graph_structure(model)
        analyze_embedding_layers(model)
        analyze_data_flow(model)
        analyze_model_capabilities(model)

        print("\n" + "=" * 80)
        print("✅ Detailed architecture analysis complete!")

    except Exception as e:
        print(f"❌ Error during detailed analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
