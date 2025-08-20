"""
AIFS Encoder Signature Analysis

Check the encoder's forward method signature to understand required arguments.
"""

import inspect
import sys
from pathlib import Path

import torch

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append("extracted_models")

from load_aifs_encoder import load_aifs_encoder


def analyze_encoder_signature():
    """Analyze the encoder's forward method signature."""
    print("🔍 Analyzing AIFS Encoder Signature")
    print("=" * 50)

    # Load encoder
    encoder, analysis = load_aifs_encoder()

    # Get forward method signature
    forward_method = encoder.forward
    signature = inspect.signature(forward_method)

    print(f"📝 Encoder Type: {type(encoder).__name__}")
    print(f"📋 Forward Method Signature:")
    print(f"   {signature}")

    print(f"\n📌 Parameters:")
    for param_name, param in signature.parameters.items():
        print(
            f"   {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}"
        )
        if param.default != inspect.Parameter.empty:
            print(f"      Default: {param.default}")

    # Try to find documentation
    if hasattr(encoder, "__doc__") and encoder.__doc__:
        print(f"\n📚 Documentation:")
        print(f"   {encoder.__doc__}")

    # Check the module
    print(f"\n🏗️ Module: {encoder.__class__.__module__}")
    print(f"📁 File: {inspect.getfile(encoder.__class__)}")

    # Look for example usage in the encoder
    print(f"\n🔍 Available Methods:")
    for method_name in dir(encoder):
        if not method_name.startswith("_") and callable(getattr(encoder, method_name)):
            method = getattr(encoder, method_name)
            if hasattr(method, "__doc__") and method.__doc__:
                print(f"   {method_name}: {method.__doc__.split('.')[0]}")
            else:
                print(f"   {method_name}")


if __name__ == "__main__":
    analyze_encoder_signature()
