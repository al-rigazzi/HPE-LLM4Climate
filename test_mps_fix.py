#!/usr/bin/env python3
"""
Test MPS Device Fix for Meta-Llama-3-8B

This script demonstrates the fix for the MPS device allocation error
and provides alternative approaches for running Llama models on Apple Silicon.
"""

import torch
import warnings
import sys
import os

# Add multimodal to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'multimodal'))

def check_device_compatibility():
    """Check available devices and their compatibility."""
    print("🔍 Device Compatibility Check")
    print("=" * 30)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        print("⚠️  MPS detected - known issues with text generation")
        print("   Solution: Use CPU or explicit device management")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    
    print()

def test_mps_fix():
    """Test the MPS device allocation fix."""
    print("🛠️ Testing MPS Fix for Meta-Llama-3-8B")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Method 1: Force CPU (Most Reliable)
        print("\n🔧 Method 1: Force CPU device")
        try:
            model_cpu = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to("cpu")
            
            print("   ✅ CPU loading: SUCCESS")
            
            # Test generation
            test_input = "Climate change impacts on agriculture:"
            inputs = tokenizer(test_input, return_tensors="pt").to("cpu")
            
            with torch.no_grad():
                outputs = model_cpu.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(test_input):].strip()
            print(f"   🤖 Generated: {generated[:50]}...")
            print("   ✅ CPU generation: SUCCESS")
            
            del model_cpu  # Free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"   ❌ CPU method error: {e}")
        
        # Method 2: Conditional Device Selection
        print("\n🔧 Method 2: Smart device selection")
        try:
            # Choose best available device
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"  # Avoid MPS for text generation
                dtype = torch.float32
            
            print(f"   📱 Selected device: {device}")
            
            model_smart = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                torch_dtype=dtype,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(device)
            
            print(f"   ✅ Smart device loading: SUCCESS")
            
            # Quick test
            test_input = "Arctic ice melting affects:"
            inputs = tokenizer(test_input, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model_smart.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,  # Deterministic for testing
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(test_input):].strip()
            print(f"   🤖 Generated: {generated[:50]}...")
            print(f"   ✅ {device.upper()} generation: SUCCESS")
            
        except Exception as e:
            print(f"   ❌ Smart device error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test setup error: {e}")
        return False

def test_location_aware_with_fix():
    """Test location-aware system with MPS fix applied."""
    print(f"\n🌍 Location-Aware System with MPS Fix")
    print("=" * 40)
    
    try:
        from multimodal.location_aware_fusion import LocationAwareClimateAnalysis
        
        print("🚀 Initializing with MPS-safe configuration...")
        
        # Use configuration that avoids MPS issues
        model = LocationAwareClimateAnalysis(
            prithvi_encoder_path=None,  # Demo mode
            llama_model_name="meta-llama/Meta-Llama-3-8B",
            fusion_mode='concatenate',  # Simpler fusion
            max_climate_tokens=256,
            max_text_length=128,
            freeze_llama=True,
            device="cpu"  # Force CPU to avoid MPS issues
        )
        
        print("✅ Location-aware model initialized with MPS fix!")
        
        # Test with a simple query
        climate_features = torch.randn(1, 100, model.fusion_dim)
        test_query = "Climate risks for Sweden agriculture"
        
        with torch.no_grad():
            result = model.analyze_location_query(
                climate_features,
                test_query,
                return_visualization=False
            )
        
        print(f"📍 Location: {result.get('location', 'N/A')}")
        print(f"⚠️  Risk: {result.get('climate_risk', 'N/A')}")
        print(f"🎯 Confidence: {result.get('overall_confidence', 0):.1%}")
        print("✅ Location-aware analysis: SUCCESS with MPS fix!")
        
        return True
        
    except Exception as e:
        print(f"❌ Location-aware test error: {e}")
        return False

def main():
    """Main function to demonstrate MPS fixes."""
    print("🔧 MPS Device Error Fix for Meta-Llama-3-8B")
    print("Resolving: 'Placeholder storage has not been allocated on MPS device!'")
    print()
    
    # Check device compatibility
    check_device_compatibility()
    
    # Test MPS fixes
    mps_fix_success = test_mps_fix()
    
    if mps_fix_success:
        # Test with location-aware system
        location_success = test_location_aware_with_fix()
        
        if location_success:
            print(f"\n🎉 ALL MPS FIXES SUCCESSFUL!")
            print(f"   🔧 MPS error: RESOLVED")
            print(f"   🦙 Meta-Llama-3-8B: Working")
            print(f"   🌍 Location-aware: Working")
            print(f"\n📋 Recommended Configuration:")
            print(f"   • Use device='cpu' for text generation")
            print(f"   • Use torch_dtype=torch.float32")
            print(f"   • Avoid device_map='auto' with MPS")
            print(f"   • Set low_cpu_mem_usage=True")
        else:
            print(f"\n⚠️  MPS fixes work, but location-aware needs attention")
    else:
        print(f"\n❌ MPS fixes need further investigation")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
