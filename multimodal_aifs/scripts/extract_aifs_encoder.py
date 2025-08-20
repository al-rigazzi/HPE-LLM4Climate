"""
AIFS Encoder Extraction

This script extracts the encoder component from the AIFS PyTorch model and saves it to disk.
The encoder is responsible for converting input atmospheric data to the model's hidden representation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from aifs_wrapper import AIFSWrapper


class AIFSEncoderExtractor:
    """
    Utility class for extracting and saving the AIFS encoder component.
    """

    def __init__(self, aifs_wrapper: AIFSWrapper):
        """
        Initialize extractor with AIFS wrapper.

        Args:
            aifs_wrapper: Initialized AIFS wrapper instance
        """
        self.aifs_wrapper = aifs_wrapper
        self.model_info = None
        self.encoder = None

    def load_model(self):
        """Load the full AIFS model."""
        print("🔄 Loading AIFS model...")
        self.model_info = self.aifs_wrapper._load_model()

        if self.model_info["pytorch_model"] is None:
            raise RuntimeError("Could not access PyTorch model for encoder extraction")

        print(
            f"✅ Successfully loaded AIFS model: {type(self.model_info['pytorch_model']).__name__}"
        )
        return self.model_info["pytorch_model"]

    def extract_encoder(self) -> nn.Module:
        """
        Extract the encoder component from the AIFS model.

        Returns:
            The encoder module
        """
        if self.model_info is None:
            full_model = self.load_model()
        else:
            full_model = self.model_info["pytorch_model"]

        # Navigate to the encoder
        # Based on our analysis: model.encoder
        if hasattr(full_model, "model") and hasattr(full_model.model, "encoder"):
            self.encoder = full_model.model.encoder
            print(f"✅ Extracted encoder: {type(self.encoder).__name__}")
        else:
            # Try alternative paths
            for name, module in full_model.named_modules():
                if name.endswith("encoder") and "GraphTransformer" in type(module).__name__:
                    self.encoder = module
                    print(f"✅ Found encoder at: {name}")
                    break

        if self.encoder is None:
            raise RuntimeError("Could not find encoder component in the model")

        return self.encoder

    def analyze_encoder(self) -> Dict[str, Any]:
        """
        Analyze the extracted encoder and return its properties.

        Returns:
            Dictionary with encoder analysis
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not extracted yet. Call extract_encoder() first.")

        # Count parameters
        total_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

        # Get encoder structure
        encoder_structure = {}
        for name, module in self.encoder.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            encoder_structure[name] = {"type": type(module).__name__, "parameters": module_params}

        # Check input/output dimensions
        input_dims = []
        output_dims = []

        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear):
                if "emb" in name and "src" in name:  # Source embedding
                    input_dims.append(module.in_features)
                    output_dims.append(module.out_features)

        analysis = {
            "encoder_type": type(self.encoder).__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "memory_mb": total_params * 4 / (1024**2),  # float32
            "structure": encoder_structure,
            "input_features": input_dims[0] if input_dims else None,
            "output_features": output_dims[0] if output_dims else None,
            "device": str(next(self.encoder.parameters()).device),
            "dtype": str(next(self.encoder.parameters()).dtype),
            "extraction_timestamp": datetime.now().isoformat(),
        }

        return analysis

    def save_encoder(
        self,
        output_dir: str = "extracted_models",
        model_name: str = "aifs_encoder",
        save_state_dict: bool = True,
        save_full_model: bool = True,
        save_analysis: bool = True,
    ) -> Dict[str, str]:
        """
        Save the extracted encoder to disk.

        Args:
            output_dir: Directory to save the encoder
            model_name: Base name for saved files
            save_state_dict: Whether to save state dict (recommended)
            save_full_model: Whether to save full model (larger file)
            save_analysis: Whether to save analysis JSON

        Returns:
            Dictionary with paths to saved files
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not extracted yet. Call extract_encoder() first.")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = {}

        # Save state dict (recommended for loading)
        if save_state_dict:
            state_dict_path = output_path / f"{model_name}_state_dict.pth"
            torch.save(self.encoder.state_dict(), state_dict_path)
            saved_files["state_dict"] = str(state_dict_path)
            print(f"✅ Saved encoder state dict: {state_dict_path}")

        # Save full model (includes architecture)
        if save_full_model:
            full_model_path = output_path / f"{model_name}_full.pth"
            torch.save(self.encoder, full_model_path)
            saved_files["full_model"] = str(full_model_path)
            print(f"✅ Saved full encoder model: {full_model_path}")

        # Save analysis
        if save_analysis:
            analysis = self.analyze_encoder()
            analysis_path = output_path / f"{model_name}_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            saved_files["analysis"] = str(analysis_path)
            print(f"✅ Saved encoder analysis: {analysis_path}")

        # Save model info
        info_path = output_path / f"{model_name}_info.txt"
        with open(info_path, "w") as f:
            f.write(f"AIFS Encoder Extraction\n")
            f.write(f"=====================\n\n")
            f.write(f"Extraction Date: {datetime.now()}\n")
            f.write(f"Source Model: AIFS Single v1.0\n")
            f.write(f"Encoder Type: {type(self.encoder).__name__}\n")
            f.write(f"Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}\n")
            f.write(f"Device: {next(self.encoder.parameters()).device}\n")
            f.write(f"Data Type: {next(self.encoder.parameters()).dtype}\n\n")

            f.write("Encoder Structure:\n")
            for name, module in self.encoder.named_children():
                params = sum(p.numel() for p in module.parameters())
                f.write(f"  {name}: {type(module).__name__} ({params:,} params)\n")

            f.write(f"\nSaved Files:\n")
            for file_type, path in saved_files.items():
                f.write(f"  {file_type}: {path}\n")

        saved_files["info"] = str(info_path)
        print(f"✅ Saved extraction info: {info_path}")

        return saved_files


def create_encoder_loader_script(
    output_dir: str = "extracted_models", model_name: str = "aifs_encoder"
):
    """
    Create a helper script for loading the extracted encoder.

    Args:
        output_dir: Directory where encoder was saved
        model_name: Base name of saved files
    """
    loader_script = f'''"""
AIFS Encoder Loader

Helper script to load the extracted AIFS encoder.
Generated automatically by encoder extraction script.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json

def load_aifs_encoder(model_dir: str = "{output_dir}",
                     model_name: str = "{model_name}",
                     device: str = "cpu") -> tuple:
    """
    Load the extracted AIFS encoder.

    Args:
        model_dir: Directory containing saved encoder
        model_name: Base name of saved files
        device: Device to load model on

    Returns:
        Tuple of (encoder_model, analysis_dict)
    """
    model_path = Path(model_dir)

    # Load analysis
    analysis_path = model_path / f"{{model_name}}_analysis.json"
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)

    # Try to load full model first
    full_model_path = model_path / f"{{model_name}}_full.pth"
    if full_model_path.exists():
        print(f"Loading full encoder model from {{full_model_path}}")
        encoder = torch.load(full_model_path, map_location=device)
        print(f"✅ Loaded encoder: {{type(encoder).__name__}}")
        print(f"   Parameters: {{sum(p.numel() for p in encoder.parameters()):,}}")
        return encoder, analysis

    # Fallback to state dict (requires architecture definition)
    state_dict_path = model_path / f"{{model_name}}_state_dict.pth"
    if state_dict_path.exists():
        print(f"State dict found at {{state_dict_path}}")
        print("⚠️  Loading state dict requires the original model architecture")
        print("   Use the full model file for easier loading")
        state_dict = torch.load(state_dict_path, map_location=device)
        return state_dict, analysis

    raise FileNotFoundError(f"No encoder files found in {{model_dir}}")

def get_encoder_info(model_dir: str = "{output_dir}", model_name: str = "{model_name}"):
    """
    Get information about the extracted encoder.

    Args:
        model_dir: Directory containing saved encoder
        model_name: Base name of saved files
    """
    model_path = Path(model_dir)
    info_path = model_path / f"{{model_name}}_info.txt"

    if info_path.exists():
        with open(info_path, 'r') as f:
            print(f.read())
    else:
        print(f"Info file not found: {{info_path}}")

if __name__ == "__main__":
    # Example usage
    try:
        print("🔄 Loading extracted AIFS encoder...")
        encoder, analysis = load_aifs_encoder()

        print(f"\\n📊 Encoder Analysis:")
        print(f"   Type: {{analysis['encoder_type']}}")
        print(f"   Parameters: {{analysis['total_parameters']:,}}")
        print(f"   Memory: {{analysis['memory_mb']:.1f}} MB")
        print(f"   Input Features: {{analysis['input_features']}}")
        print(f"   Output Features: {{analysis['output_features']}}")

        print(f"\\n🏗️  Encoder Structure:")
        for name, info in analysis['structure'].items():
            print(f"   {{name}}: {{info['type']}} ({{info['parameters']:,}} params)")

        print(f"\\n✅ Encoder ready for use!")

    except Exception as e:
        print(f"❌ Error loading encoder: {{e}}")
        print("\\nℹ️  Encoder information:")
        get_encoder_info()
'''

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    loader_path = output_path / "load_aifs_encoder.py"
    with open(loader_path, "w") as f:
        f.write(loader_script)

    print(f"✅ Created encoder loader script: {loader_path}")
    return str(loader_path)


def main():
    """Main function for encoder extraction."""
    try:
        print("🚀 AIFS Encoder Extraction")
        print("=" * 50)

        # Initialize AIFS wrapper
        print("🔧 Initializing AIFS wrapper...")
        aifs = AIFSWrapper()

        # Create extractor
        extractor = AIFSEncoderExtractor(aifs)

        # Extract encoder
        print("\\n🎯 Extracting encoder component...")
        encoder = extractor.extract_encoder()

        # Analyze encoder
        print("\\n🔍 Analyzing encoder...")
        analysis = extractor.analyze_encoder()

        print(f"\\n📊 Encoder Analysis:")
        print(f"   Type: {analysis['encoder_type']}")
        print(f"   Total Parameters: {analysis['total_parameters']:,}")
        print(f"   Trainable Parameters: {analysis['trainable_parameters']:,}")
        print(f"   Memory Size: {analysis['memory_mb']:.1f} MB")
        print(f"   Input Features: {analysis['input_features']}")
        print(f"   Output Features: {analysis['output_features']}")

        print(f"\\n🏗️  Encoder Structure:")
        for name, info in analysis["structure"].items():
            print(f"   {name:20} {info['type']:25} {info['parameters']:>12,} params")

        # Save encoder
        print("\\n💾 Saving encoder to disk...")
        saved_files = extractor.save_encoder(
            output_dir="extracted_models",
            model_name="aifs_encoder",
            save_state_dict=True,
            save_full_model=True,
            save_analysis=True,
        )

        print(f"\\n📁 Saved Files:")
        for file_type, path in saved_files.items():
            file_size = Path(path).stat().st_size / (1024**2)  # MB
            print(f"   {file_type:12} {path} ({file_size:.1f} MB)")

        # Create loader script
        print("\\n📝 Creating loader script...")
        loader_script = create_encoder_loader_script("extracted_models", "aifs_encoder")

        print(f"\\n✅ Encoder extraction complete!")
        print(f"\\n🎯 Next Steps:")
        print(f"   1. Use the extracted encoder in your own projects")
        print(f"   2. Load with: python extracted_models/load_aifs_encoder.py")
        print(f"   3. Or import the loader functions in your code")

        print(f"\\n📚 Usage Example:")
        print(f"   from extracted_models.load_aifs_encoder import load_aifs_encoder")
        print(f"   encoder, analysis = load_aifs_encoder()")

    except Exception as e:
        print(f"❌ Error during encoder extraction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
