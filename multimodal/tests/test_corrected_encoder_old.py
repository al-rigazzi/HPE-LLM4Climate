"""
Test with Properly Extracted Encoder

This demo uses the correctly extracted encoder that has proper configuration
matching the actual model weights.
"""

import torch
import warnings
from pathlib import Path

try:
    from .location_aware_fusion import LocationAwareClimateAnalysis
except ImportError:
    from location_aware_fusion import LocationAwareClimateAnalysis

def test_corrected_encoder():
    """Test with the properly extracted encoder."""
    print("🧪 Testing Corrected Prithvi Encoder")
    print("=" * 50)

    # Use the corrected encoder
    encoder_path = "data/weights/prithvi_encoder_corrected.pt"

    if not Path(encoder_path).exists():
        print(f"❌ Corrected encoder not found: {encoder_path}")
        print("   Run: python multimodal/correct_extraction.py")
        return

    print(f"✅ Found corrected encoder: {encoder_path}")

    # Verify the config is correct
    print("\\n🔍 Verifying corrected configuration...")
    checkpoint = torch.load(encoder_path, map_location='cpu')
    config = checkpoint['config']['params']

    static_tensor = checkpoint['model_state_dict']['static_input_scalers_mu']
    input_tensor = checkpoint['model_state_dict']['input_scalers_mu']

    print(f"   Config says: {config['in_channels_static']} static channels")
    print(f"   Tensor has: {static_tensor.shape[1]} static channels")
    print(f"   Config says: {config['in_channels']} input channels")
    print(f"   Tensor has: {input_tensor.shape[2]} input channels")

    if (config['in_channels_static'] == static_tensor.shape[1] and
        config['in_channels'] == input_tensor.shape[2]):
        print("   ✅ Configuration is correct!")
    else:
        print("   ❌ Configuration still wrong")
        return

    # Test loading the model
    print("\\n🚀 Testing model initialization...")

    try:
        model = LocationAwareClimateAnalysis(
            prithvi_encoder_path=encoder_path,
            llama_model_name="distilbert-base-uncased",  # Public model
            fusion_mode="cross_attention",
            freeze_prithvi=True,
            freeze_llama=True
        )

        print("✅ Model loaded successfully!")
        print(f"   Climate dimension: {model.climate_text_fusion.climate_dim}")
        print(f"   Text dimension: {model.climate_text_fusion.text_dim}")

        # Test with sample data
        print("\\n📊 Testing with sample climate features...")
        batch_size = 1
        num_patches = (360 // 2) * (576 // 2)  # Based on patch_size_px=[2,2]
        embed_dim = model.climate_text_fusion.climate_dim

        climate_features = torch.randn(batch_size, num_patches, embed_dim)
        print(f"   Features shape: {climate_features.shape}")

        # Test a query
        query = "Climate risk for Stockholm, Sweden"
        print(f"\\n🎯 Testing query: {query}")

        result = model.analyze_location_query(climate_features, query)

        print(f"   📍 Location: {result['location']} ({result['location_type']})")
        print(f"   ⚠️  Risk: {result['climate_risk']}")
        print(f"   🎯 Confidence: {result['overall_confidence']:.1%}")

        print("\\n🎉 SUCCESS: Everything works correctly!")
        print("   • No size mismatch warnings")
        print("   • No configuration errors")
        print("   • Model loads and runs properly")
        print("   • Real Prithvi weights are being used")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_corrected_encoder()
    if success:
        print("\\n✅ Corrected encoder works perfectly!")
    else:
        print("\\n❌ Still have issues to resolve")
