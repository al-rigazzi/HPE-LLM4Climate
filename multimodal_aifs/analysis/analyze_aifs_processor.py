"""
AIFS Processor Analysis

Deep dive into the Processor component of the AIFS model - the core transformer that handles
the main computation and contains the majority of the model's parameters.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from aifs_wrapper import AIFSWrapper


def analyze_processor_architecture(processor: nn.Module) -> Dict[str, Any]:
    """
    Analyze the processor architecture in detail.

    Args:
        processor: The processor module

    Returns:
        Dictionary with detailed processor analysis
    """
    total_params = sum(p.numel() for p in processor.parameters())

    analysis = {
        "processor_type": type(processor).__name__,
        "total_parameters": total_params,
        "memory_mb": total_params * 4 / (1024**2),
        "structure": {},
        "transformer_layers": [],
        "attention_patterns": {},
        "mlp_patterns": {},
    }

    # Analyze top-level structure
    for name, module in processor.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        analysis["structure"][name] = {
            "type": type(module).__name__,
            "parameters": module_params,
            "percentage": (module_params / total_params) * 100 if total_params > 0 else 0,
        }

    # Analyze transformer layers if they exist
    if hasattr(processor, "proc") and hasattr(processor.proc, "__iter__"):
        for layer_idx, layer in enumerate(processor.proc):
            layer_info = analyze_transformer_layer(layer, layer_idx)
            analysis["transformer_layers"].append(layer_info)

    # Analyze attention patterns
    attention_info = analyze_attention_patterns(processor)
    analysis["attention_patterns"] = attention_info

    # Analyze MLP patterns
    mlp_info = analyze_mlp_patterns(processor)
    analysis["mlp_patterns"] = mlp_info

    return analysis


def analyze_transformer_layer(layer: nn.Module, layer_idx: int) -> Dict[str, Any]:
    """
    Analyze a single transformer layer in detail.

    Args:
        layer: The transformer layer module
        layer_idx: Index of the layer

    Returns:
        Dictionary with layer analysis
    """
    layer_params = sum(p.numel() for p in layer.parameters())

    layer_info = {
        "layer_index": layer_idx,
        "layer_type": type(layer).__name__,
        "total_parameters": layer_params,
        "blocks": [],
        "layer_structure": {},
    }

    # Analyze layer structure
    for name, module in layer.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        layer_info["layer_structure"][name] = {
            "type": type(module).__name__,
            "parameters": module_params,
        }

    # Analyze transformer blocks within the layer
    if hasattr(layer, "blocks"):
        for block_idx, block in enumerate(layer.blocks):
            block_info = analyze_transformer_block(block, block_idx)
            layer_info["blocks"].append(block_info)

    return layer_info


def analyze_transformer_block(block: nn.Module, block_idx: int) -> Dict[str, Any]:
    """
    Analyze a single transformer block in detail.

    Args:
        block: The transformer block module
        block_idx: Index of the block

    Returns:
        Dictionary with block analysis
    """
    block_params = sum(p.numel() for p in block.parameters())

    block_info = {
        "block_index": block_idx,
        "block_type": type(block).__name__,
        "total_parameters": block_params,
        "components": {},
    }

    # Analyze each component in the block
    for name, component in block.named_children():
        comp_params = sum(p.numel() for p in component.parameters())
        comp_info = {
            "type": type(component).__name__,
            "parameters": comp_params,
            "percentage": (comp_params / block_params) * 100 if block_params > 0 else 0,
        }

        # Add specific details for different component types
        if "attention" in name.lower():
            if hasattr(component, "num_heads"):
                comp_info["num_heads"] = component.num_heads
            if hasattr(component, "embed_dim"):
                comp_info["embed_dim"] = component.embed_dim
            if hasattr(component, "head_dim"):
                comp_info["head_dim"] = component.head_dim

        elif "mlp" in name.lower() or isinstance(component, nn.Sequential):
            # Analyze MLP structure
            mlp_layers = []
            for mlp_name, mlp_layer in component.named_children():
                if isinstance(mlp_layer, nn.Linear):
                    mlp_layers.append(
                        {
                            "name": mlp_name,
                            "type": type(mlp_layer).__name__,
                            "in_features": mlp_layer.in_features,
                            "out_features": mlp_layer.out_features,
                            "parameters": sum(p.numel() for p in mlp_layer.parameters()),
                        }
                    )
            comp_info["mlp_layers"] = mlp_layers

        elif isinstance(component, nn.LayerNorm):
            comp_info["normalized_shape"] = component.normalized_shape

        block_info["components"][name] = comp_info

    return block_info


def analyze_attention_patterns(processor: nn.Module) -> Dict[str, Any]:
    """
    Analyze attention patterns across the processor.

    Args:
        processor: The processor module

    Returns:
        Dictionary with attention analysis
    """
    attention_layers = []

    for name, module in processor.named_modules():
        if "attention" in name.lower() or "attn" in name.lower():
            if hasattr(module, "num_heads") or hasattr(module, "embed_dim"):
                attention_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                }

                for attr in ["num_heads", "embed_dim", "head_dim", "dropout"]:
                    if hasattr(module, attr):
                        attention_info[attr] = getattr(module, attr)

                attention_layers.append(attention_info)

    # Calculate statistics
    if attention_layers:
        total_attention_params = sum(layer["parameters"] for layer in attention_layers)
        avg_attention_params = total_attention_params / len(attention_layers)

        return {
            "total_attention_layers": len(attention_layers),
            "total_attention_parameters": total_attention_params,
            "average_parameters_per_layer": avg_attention_params,
            "attention_layers": attention_layers,
        }

    return {"total_attention_layers": 0}


def analyze_mlp_patterns(processor: nn.Module) -> Dict[str, Any]:
    """
    Analyze MLP patterns across the processor.

    Args:
        processor: The processor module

    Returns:
        Dictionary with MLP analysis
    """
    mlp_layers = []

    for name, module in processor.named_modules():
        if "mlp" in name.lower() and isinstance(module, nn.Sequential):
            mlp_info = {
                "name": name,
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "layers": [],
            }

            for layer_name, layer in module.named_children():
                if isinstance(layer, nn.Linear):
                    mlp_info["layers"].append(
                        {
                            "name": layer_name,
                            "in_features": layer.in_features,
                            "out_features": layer.out_features,
                            "parameters": sum(p.numel() for p in layer.parameters()),
                        }
                    )

            mlp_layers.append(mlp_info)

    # Calculate statistics
    if mlp_layers:
        total_mlp_params = sum(layer["parameters"] for layer in mlp_layers)
        avg_mlp_params = total_mlp_params / len(mlp_layers)

        # Find expansion patterns
        expansion_ratios = []
        for mlp in mlp_layers:
            if len(mlp["layers"]) >= 2:
                first_layer = mlp["layers"][0]
                expansion_ratio = first_layer["out_features"] / first_layer["in_features"]
                expansion_ratios.append(expansion_ratio)

        return {
            "total_mlp_layers": len(mlp_layers),
            "total_mlp_parameters": total_mlp_params,
            "average_parameters_per_layer": avg_mlp_params,
            "expansion_ratios": expansion_ratios,
            "average_expansion_ratio": (
                sum(expansion_ratios) / len(expansion_ratios) if expansion_ratios else 0
            ),
            "mlp_layers": mlp_layers,
        }

    return {"total_mlp_layers": 0}


def print_processor_summary(analysis: Dict[str, Any]):
    """
    Print a comprehensive summary of the processor analysis.

    Args:
        analysis: Analysis dictionary from analyze_processor_architecture
    """
    print("=" * 80)
    print("🔧 AIFS Processor Detailed Analysis")
    print("=" * 80)

    # Basic information
    print(f"\n📊 Processor Overview:")
    print(f"   Type: {analysis['processor_type']}")
    print(f"   Total Parameters: {analysis['total_parameters']:,}")
    print(f"   Memory Size: {analysis['memory_mb']:.1f} MB")
    print(f"   Percentage of Full Model: {(analysis['total_parameters'] / 253035398) * 100:.1f}%")

    # Structure breakdown
    print(f"\n🏗️ Processor Structure:")
    for name, info in analysis["structure"].items():
        print(
            f"   {name:15} {info['type']:25} {info['parameters']:>12,} params ({info['percentage']:5.1f}%)"
        )

    # Transformer layers
    if analysis["transformer_layers"]:
        print(f"\n🧱 Transformer Layers ({len(analysis['transformer_layers'])}):")
        for layer in analysis["transformer_layers"]:
            print(
                f"   Layer {layer['layer_index']:2d}: {layer['layer_type']:25} {layer['total_parameters']:>12,} params"
            )

            if layer["blocks"]:
                print(f"      └─ Transformer Blocks: {len(layer['blocks'])}")
                for block in layer["blocks"][:3]:  # Show first 3 blocks
                    print(
                        f"         Block {block['block_index']:2d}: {block['total_parameters']:>10,} params"
                    )
                    for comp_name, comp_info in block["components"].items():
                        print(
                            f"            {comp_name:12} {comp_info['type']:20} {comp_info['parameters']:>8,} params ({comp_info['percentage']:4.1f}%)"
                        )

                if len(layer["blocks"]) > 3:
                    print(f"         ... and {len(layer['blocks']) - 3} more blocks")

    # Attention analysis
    if analysis["attention_patterns"].get("total_attention_layers", 0) > 0:
        attn = analysis["attention_patterns"]
        print(f"\n🎯 Attention Analysis:")
        print(f"   Total Attention Layers: {attn['total_attention_layers']}")
        print(f"   Total Attention Parameters: {attn['total_attention_parameters']:,}")
        print(f"   Average Parameters per Layer: {attn['average_parameters_per_layer']:,.0f}")

        # Show details for first few attention layers
        print(f"\n   Attention Layer Details:")
        for attn_layer in attn["attention_layers"][:5]:  # Show first 5
            details = []
            if "num_heads" in attn_layer:
                details.append(f"heads:{attn_layer['num_heads']}")
            if "embed_dim" in attn_layer:
                details.append(f"dim:{attn_layer['embed_dim']}")
            if "head_dim" in attn_layer:
                details.append(f"head_dim:{attn_layer['head_dim']}")

            detail_str = ", ".join(details)
            print(
                f"      {attn_layer['name']:40} {attn_layer['parameters']:>10,} params ({detail_str})"
            )

        if len(attn["attention_layers"]) > 5:
            print(f"      ... and {len(attn['attention_layers']) - 5} more attention layers")

    # MLP analysis
    if analysis["mlp_patterns"].get("total_mlp_layers", 0) > 0:
        mlp = analysis["mlp_patterns"]
        print(f"\n🔗 MLP Analysis:")
        print(f"   Total MLP Layers: {mlp['total_mlp_layers']}")
        print(f"   Total MLP Parameters: {mlp['total_mlp_parameters']:,}")
        print(f"   Average Parameters per Layer: {mlp['average_parameters_per_layer']:,.0f}")
        print(f"   Average Expansion Ratio: {mlp['average_expansion_ratio']:.1f}x")

        # Show MLP architecture pattern
        if mlp["mlp_layers"]:
            sample_mlp = mlp["mlp_layers"][0]
            print(f"\n   MLP Architecture Pattern (from {sample_mlp['name']}):")
            for layer in sample_mlp["layers"]:
                print(
                    f"      {layer['name']:10} {layer['in_features']:>5} → {layer['out_features']:>5} features ({layer['parameters']:>10,} params)"
                )


def main():
    """Main function for processor analysis."""
    try:
        print("🔬 AIFS Processor Deep Dive Analysis")
        print("=" * 50)

        # Initialize and load model
        aifs = AIFSWrapper()
        model_info = aifs._load_model()

        if model_info["pytorch_model"] is None:
            print("❌ Could not access PyTorch model")
            return

        full_model = model_info["pytorch_model"]

        # Find the processor
        processor = None
        for name, module in full_model.named_modules():
            if name.endswith("processor") and "Transformer" in type(module).__name__:
                processor = module
                print(f"✅ Found processor: {name} ({type(module).__name__})")
                break

        if processor is None:
            print("❌ Could not find processor component")
            return

        # Analyze processor
        print(f"\n🔍 Analyzing processor architecture...")
        analysis = analyze_processor_architecture(processor)

        # Print comprehensive summary
        print_processor_summary(analysis)

        # Save analysis to file
        output_file = "processor_analysis.json"
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Detailed analysis saved to: {output_file}")

        print(f"\n" + "=" * 80)
        print("✅ Processor analysis complete!")

    except Exception as e:
        print(f"❌ Error during processor analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
