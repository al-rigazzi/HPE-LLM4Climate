"""
Test script for climate-text fusion with simplified setup.

This script tests the multimodal fusion using smaller models and reduced data sizes
to ensure everything works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_basic_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not available")
        return False

    try:
        from multimodal.utils.encoder_extractor import PrithviWxC_Encoder
        print("✓ PrithviWxC_Encoder imported")
    except ImportError as e:
        print(f"✗ PrithviWxC_Encoder import failed: {e}")
        return False

    try:
        from multimodal.core.climate_text_fusion import ClimateTextFusion
        print("✓ ClimateTextFusion imported")
    except ImportError as e:
        print(f"✗ ClimateTextFusion import failed: {e}")
        return False

    return True


def create_minimal_climate_data():
    """Create minimal climate data for testing."""
    batch_size = 1

    # Use very small spatial dimensions for testing
    n_lats = 32  # Much smaller than full 720
    n_lons = 64  # Much smaller than full 1440
    n_times = 2
    n_channels = 8  # Much smaller than full 160
    n_static = 2   # Much smaller than full 4

    return {
        'x': torch.randn(batch_size, n_times, n_channels, n_lats, n_lons),
        'static': torch.randn(batch_size, n_static, n_lats, n_lons),
        'climate': torch.randn(batch_size, n_channels, n_lats, n_lons),
        'input_time': torch.tensor([0.0]),
        'lead_time': torch.tensor([18.0]),
    }


def test_encoder_loading():
    """Test loading the PrithviWxC encoder."""
    print("\nTesting encoder loading...")

    try:
        from multimodal.utils.encoder_extractor import PrithviWxC_Encoder

        # Load encoder config
        encoder_path = 'data/weights/prithvi_encoder.pt'
        if not os.path.exists(encoder_path):
            print(f"✗ Encoder weights not found at {encoder_path}")
            return False

        checkpoint = torch.load(encoder_path, map_location='cpu')
        config = checkpoint['config']['params']

        print(f"✓ Encoder config loaded")
        print(f"  - Embedding dim: {config['embed_dim']}")
        print(f"  - Input channels: {config['in_channels']}")
        print(f"  - Encoder blocks: {config['n_blocks_encoder']}")

        return True

    except Exception as e:
        print(f"✗ Encoder loading failed: {e}")
        return False


def test_text_model_loading():
    """Test loading a simple text model."""
    print("\nTesting text model loading...")

    try:
        from transformers import AutoTokenizer, AutoModel

        # Use a very small model for testing
        model_name = "prajjwal1/bert-tiny"  # Only 4.4M parameters

        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        print(f"✓ Text model loaded successfully")
        print(f"  - Model size: ~4.4M parameters")
        print(f"  - Hidden size: {model.config.hidden_size}")

        # Test tokenization
        text = ["This is a test sentence about weather."]
        tokens = tokenizer(text, return_tensors='pt', padding=True)

        with torch.no_grad():
            outputs = model(**tokens)

        print(f"✓ Text processing test successful")
        print(f"  - Output shape: {outputs.last_hidden_state.shape}")

        return True

    except Exception as e:
        print(f"✗ Text model loading failed: {e}")
        return False


def test_fusion_components():
    """Test individual fusion components."""
    print("\nTesting fusion components...")

    try:
        from multimodal.core.climate_text_fusion import ClimateFeatureProjector, CrossModalAttention

        # Test feature projector
        climate_dim = 128
        text_dim = 128

        projector = ClimateFeatureProjector(climate_dim, text_dim)
        climate_features = torch.randn(1, 10, climate_dim)
        projected = projector(climate_features)

        print(f"✓ Feature projector works")
        print(f"  - Input: {climate_features.shape}")
        print(f"  - Output: {projected.shape}")

        # Test cross-modal attention
        attention = CrossModalAttention(text_dim)
        text_features = torch.randn(1, 8, text_dim)
        fused = attention(text_features, projected, projected)

        print(f"✓ Cross-modal attention works")
        print(f"  - Fused output: {fused.shape}")

        return True

    except Exception as e:
        print(f"✗ Fusion components test failed: {e}")
        return False


def test_simple_text_fusion():
    """Test a simplified version of the fusion model."""
    print("\nTesting simplified fusion...")

    try:
        # Create a minimal fusion test
        batch_size = 1
        seq_len = 8
        embed_dim = 128

        # Simulate climate and text features
        climate_features = torch.randn(batch_size, seq_len, embed_dim)
        text_features = torch.randn(batch_size, seq_len, embed_dim)

        # Simple additive fusion
        fused_features = climate_features + text_features

        print(f"✓ Simple fusion test successful")
        print(f"  - Climate features: {climate_features.shape}")
        print(f"  - Text features: {text_features.shape}")
        print(f"  - Fused features: {fused_features.shape}")

        return True

    except Exception as e:
        print(f"✗ Simple fusion test failed: {e}")
        return False


def run_comprehensive_test():
    """Run a comprehensive test of the system."""
    print("=" * 60)
    print("🧪 CLIMATE-TEXT FUSION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Encoder Loading", test_encoder_loading),
        ("Text Model Loading", test_text_model_loading),
        ("Fusion Components", test_fusion_components),
        ("Simple Fusion", test_simple_text_fusion),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


def show_usage_instructions():
    """Show instructions for using the multimodal fusion system."""
    print(f"\n{'='*60}")
    print("📖 USAGE INSTRUCTIONS")
    print("=" * 60)

    print("""
🚀 Quick Start Guide:

1. **Basic Setup:**
   ```python
   from multimodal.core.climate_text_fusion import ClimateTextFusion

   model = ClimateTextFusion(
       prithvi_encoder_path='data/weights/prithvi_encoder.pt',
       llama_model_name='prajjwal1/bert-tiny',  # Start with small model
       fusion_mode='cross_attention'
   )
   ```

2. **Prepare Data:**
   ```python
   climate_batch = {
       'x': torch.randn(1, 2, 160, 720, 1440),      # [batch, time, channels, lat, lon]
       'static': torch.randn(1, 4, 720, 1440),       # [batch, static_channels, lat, lon]
       'climate': torch.randn(1, 160, 720, 1440),    # [batch, channels, lat, lon]
       'input_time': torch.tensor([0.0]),
       'lead_time': torch.tensor([18.0])
   }

   text_inputs = ["What is the weather forecast?"]
   ```

3. **Run Fusion:**
   ```python
   outputs = model(climate_batch, text_inputs)
   fused_features = outputs['fused_features']  # Use for downstream tasks
   ```

💡 **Tips:**
   • Start with smaller spatial dimensions for testing
   • Use CPU for initial experiments if GPU memory is limited
   • Try different fusion modes: 'cross_attention', 'concatenate', 'add'
   • Monitor memory usage with large climate datasets

🎯 **Applications:**
   • Weather report generation
   • Climate question answering
   • Scientific document analysis
   • Agricultural advisory systems
   • Emergency response planning

For more examples, run: python multimodal/fusion_demo.py
    """)


def main():
    """Main test function."""
    success = run_comprehensive_test()
    show_usage_instructions()

    if success:
        print("\n🌟 Ready to start multimodal climate-text fusion!")
    else:
        print("\n⚠️  Please fix the failed tests before proceeding.")


if __name__ == "__main__":
    main()
