"""
AIFS Encoder Loader

Helper script to load the extracted AIFS encoder.
Generated automatically by encoder extraction script.
"""

import json
from pathlib import Path

import torch


def load_aifs_encoder(
    model_dir: str = None, model_name: str = "aifs_encoder", device: str = "cpu"
) -> tuple:
    """
    Load the extracted AIFS encoder.

    Args:
        model_dir: Directory containing saved encoder (if None, uses current file's directory)
        model_name: Base name of saved files
        device: Device to load model on

    Returns:
        tuple of (encoder_model, analysis_dict)
    """
    if model_dir is None:
        # Use the directory where this script is located
        model_path = Path(__file__).parent
    else:
        model_path = Path(model_dir)  # Load analysis
    analysis_path = model_path / f"{model_name}_analysis.json"
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    # Try to load full model first
    full_model_path = model_path / f"{model_name}_full.pth"
    if full_model_path.exists():
        print(f"Loading full encoder model from {full_model_path}")
        encoder = torch.load(full_model_path, map_location=device)
        print(f"Loaded encoder: {type(encoder).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        return encoder, analysis

    # Fallback to state dict (requires architecture definition)
    state_dict_path = model_path / f"{model_name}_state_dict.pth"
    if state_dict_path.exists():
        print(f"State dict found at {state_dict_path}")
        print("Loading state dict requires the original model architecture")
        print("   Use the full model file for easier loading")
        state_dict = torch.load(state_dict_path, map_location=device)
        return state_dict, analysis

    raise FileNotFoundError(f"No encoder files found in {model_dir}")


def get_encoder_info(model_dir: str = None, model_name: str = "aifs_encoder"):
    """
    Get information about the extracted encoder.

    Args:
        model_dir: Directory containing saved encoder (if None, uses current file's directory)
        model_name: Base name of saved files
    """
    if model_dir is None:
        model_path = Path(__file__).parent
    else:
        model_path = Path(model_dir)
    info_path = model_path / f"{model_name}_info.txt"

    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print(f"Info file not found: {info_path}")


if __name__ == "__main__":
    # Example usage
    try:
        print("Loading extracted AIFS encoder...")
        encoder, analysis = load_aifs_encoder()

        print(f"\nEncoder Analysis:")
        print(f"   Type: {analysis['encoder_type']}")
        print(f"   Parameters: {analysis['total_parameters']:,}")
        print(f"   Memory: {analysis['memory_mb']:.1f} MB")
        print(f"   Input Features: {analysis['input_features']}")
        print(f"   Output Features: {analysis['output_features']}")

        print(f"\n🏗️  Encoder Structure:")
        for name, info in analysis["structure"].items():
            print(f"   {name}: {info['type']} ({info['parameters']:,} params)")

        print(f"\nEncoder ready for use!")

    except Exception as e:
        print(f"Error loading encoder: {e}")
        print("\nℹ️  Encoder information:")
        get_encoder_info()
