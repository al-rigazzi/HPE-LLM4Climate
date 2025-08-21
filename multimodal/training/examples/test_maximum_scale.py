#!/usr/bin/env python3
"""
Maximum Scale Test - Simulating Llama-3-8B Scale Training

This script tests the limits of our 36GB RAM by loading the largest available
models and simulating what Llama-3-8B training would require.
"""

import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Memory optimization
torch.set_num_threads(1)  # Single thread for maximum memory
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

print("🚀 Maximum Scale Test - Pushing 36GB RAM Limits")


def check_memory_usage():
    """Check current memory usage"""
    import psutil

    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024**3
    return memory_gb


def get_system_memory():
    """Get total system memory"""
    import psutil

    return psutil.virtual_memory().total / 1024**3


def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()


# Check system specs
total_ram = get_system_memory()
print(f"💾 Total System RAM: {total_ram:.1f} GB")
print(f"🎯 Target: Use as much RAM as safely possible")

# Try to load multiple large models or the largest available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("✅ Transformers available")

    # Try progressively larger models
    model_candidates = [
        ("microsoft/DialoGPT-large", "DialoGPT-Large", "762M"),
        ("gpt2-xl", "GPT-2 XL", "1.5B"),  # This is much larger!
        ("EleutherAI/gpt-neo-1.3B", "GPT-Neo 1.3B", "1.3B"),
    ]

    loaded_models = []
    total_params = 0

    for model_name, description, size in model_candidates:
        try:
            print(f"\n🔍 Attempting to load: {description} ({size})")
            memory_before = check_memory_usage()
            print(f"💾 RAM before: {memory_before:.2f} GB")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with memory optimization
            print("📥 Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                use_cache=False,
            )

            params = sum(p.numel() for p in model.parameters())
            total_params += params

            memory_after = check_memory_usage()
            memory_used = memory_after - memory_before

            print(f"✅ Loaded {description}: {params:,} parameters")
            print(f"💾 RAM after: {memory_after:.2f} GB (+{memory_used:.2f} GB)")

            loaded_models.append((model, description, params))

            # Check if we're approaching memory limits
            if memory_after > 30:  # Leave 6GB buffer
                print(f"⚠️ Approaching memory limit, stopping model loading")
                break

        except Exception as e:
            print(f"❌ Failed to load {description}: {e}")
            clear_memory()
            continue

    if not loaded_models:
        print("❌ Could not load any large models")
        sys.exit(1)

    print(f"\n📊 Successfully loaded {len(loaded_models)} large models")
    print(f"📊 Total parameters across all models: {total_params:,}")
    print(f"💾 Current RAM usage: {check_memory_usage():.2f} GB")

except Exception as e:
    print(f"❌ Error in model loading phase: {e}")
    sys.exit(1)

# Create a fusion model using the largest loaded model
print(f"\n🏗️ Creating maximum-scale fusion model...")

# Use the largest model
largest_model = max(loaded_models, key=lambda x: x[2])
base_model, model_desc, model_params = largest_model

print(f"🎯 Using {model_desc} as base ({model_params:,} parameters)")


class MaxScaleFusion(torch.nn.Module):
    def __init__(self, text_model, climate_dim=512):  # Larger climate features
        super().__init__()

        # Larger climate encoder to simulate real-world complexity
        self.climate_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(20, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 8 * 8, climate_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(climate_dim, climate_dim),
        )

        self.text_model = text_model
        self.text_hidden_size = text_model.config.hidden_size

        # Larger projection network
        self.climate_projector = torch.nn.Sequential(
            torch.nn.Linear(climate_dim, self.text_hidden_size),
            torch.nn.LayerNorm(self.text_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.text_hidden_size, self.text_hidden_size),
            torch.nn.LayerNorm(self.text_hidden_size),
        )

        # Multi-layer cross attention
        self.fusion_layers = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    embed_dim=self.text_hidden_size, num_heads=8, batch_first=True
                )
                for _ in range(4)  # Multiple fusion layers
            ]
        )

        # Layer norms for each fusion layer
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(self.text_hidden_size) for _ in range(4)]
        )

        # Output projection
        self.output_projection = torch.nn.Linear(self.text_hidden_size, 2000)  # Larger vocab

        # Freeze text model
        for param in self.text_model.parameters():
            param.requires_grad = False

        print(f"📊 Text model frozen ({self.text_hidden_size} hidden size)")

    def forward(self, climate_data, input_ids):
        # Process climate data for each timestep
        batch_size, time_steps = climate_data.shape[:2]

        climate_outputs = []
        for t in range(time_steps):
            climate_t = self.climate_encoder(climate_data[:, t])
            climate_outputs.append(climate_t)

        climate_features = torch.stack(climate_outputs, dim=1)
        projected_climate = self.climate_projector(climate_features)

        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_model.get_input_embeddings()(input_ids)

        # Multi-layer fusion
        fused_features = text_embeddings
        for i, (attention_layer, norm_layer) in enumerate(
            zip(self.fusion_layers, self.layer_norms)
        ):
            # Cross attention
            attended_features, _ = attention_layer(
                query=fused_features, key=projected_climate, value=projected_climate
            )

            # Residual connection and normalization
            fused_features = norm_layer(fused_features + attended_features)

        # Output projection
        logits = self.output_projection(fused_features)

        return type("Output", (), {"logits": logits})()


def main():
    memory_before_fusion = check_memory_usage()
    print(f"💾 RAM before fusion model: {memory_before_fusion:.2f} GB")

    # Create maximum scale fusion model
    try:
        fusion_model = MaxScaleFusion(base_model, climate_dim=512)

        total_params = sum(p.numel() for p in fusion_model.parameters())
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)

        print(f"📊 Total fusion parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📈 Frozen: {((total_params - trainable_params) / total_params * 100):.1f}%")

        memory_after_fusion = check_memory_usage()
        print(f"💾 RAM after fusion model: {memory_after_fusion:.2f} GB")
        print(f"📈 Fusion overhead: {memory_after_fusion - memory_before_fusion:.2f} GB")

    except Exception as e:
        print(f"❌ Failed to create fusion model: {e}")
        return

    # Create realistic-sized data
    print(f"\n📊 Creating larger-scale test data...")
    batch_size = 1  # Keep small for memory
    seq_length = 64  # Longer sequences
    time_steps = 4  # More timesteps

    # Larger climate data
    climate_data = torch.randn(batch_size, time_steps, 20, 16, 16)
    input_ids = torch.randint(1, 2000, (batch_size, seq_length))
    labels = torch.randint(0, 2000, (batch_size, seq_length))

    print(f"Climate shape: {climate_data.shape}")
    print(f"Text shape: {input_ids.shape}")

    # Test forward pass
    print(f"\n🚀 Testing large-scale forward pass...")
    memory_before_forward = check_memory_usage()

    try:
        fusion_model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = fusion_model(climate_data, input_ids)
            forward_time = time.time() - start_time

        print(f"✅ Forward pass successful!")
        print(f"📊 Output shape: {outputs.logits.shape}")
        print(f"⚡ Forward time: {forward_time:.2f} seconds")

        memory_after_forward = check_memory_usage()
        print(f"💾 RAM after forward: {memory_after_forward:.2f} GB")
        print(f"📈 Forward memory delta: {memory_after_forward - memory_before_forward:.2f} GB")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # Test training with memory monitoring
    print(f"\n🎯 Testing large-scale training...")
    memory_before_train = check_memory_usage()

    try:
        fusion_model.train()

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in fusion_model.parameters() if p.requires_grad], lr=5e-5
        )

        # Training step with timing
        start_time = time.time()

        outputs = fusion_model(climate_data, input_ids)

        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
        )

        print(f"📊 Loss: {loss.item():.4f}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in fusion_model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()
        optimizer.zero_grad()

        train_time = time.time() - start_time

        memory_after_train = check_memory_usage()
        print(f"✅ Training step successful!")
        print(f"⚡ Training time: {train_time:.2f} seconds")
        print(f"🔧 Gradient norm: {grad_norm:.4f}")
        print(f"💾 RAM after training: {memory_after_train:.2f} GB")
        print(f"📈 Training memory delta: {memory_after_train - memory_before_train:.2f} GB")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # Memory stress test - run multiple training steps
    print(f"\n🔥 Memory stress test - multiple training steps...")

    try:
        peak_memory = memory_after_train

        for step in range(5):
            step_start = time.time()

            outputs = fusion_model(climate_data, input_ids)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in fusion_model.parameters() if p.requires_grad], max_norm=1.0
            )

            optimizer.step()

            step_time = time.time() - step_start
            current_memory = check_memory_usage()
            peak_memory = max(peak_memory, current_memory)

            print(
                f"Step {step+1}: Loss={loss.item():.4f}, Time={step_time:.2f}s, RAM={current_memory:.1f}GB"
            )

            clear_memory()

        print(f"✅ Stress test completed!")
        print(f"📊 Peak memory usage: {peak_memory:.2f} GB")

    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return

    final_memory = check_memory_usage()
    memory_efficiency = (final_memory / total_ram) * 100

    print(f"\n🎉 Maximum scale test completed!")
    print(f"💾 Final RAM usage: {final_memory:.2f} GB / {total_ram:.1f} GB")
    print(f"📊 Memory efficiency: {memory_efficiency:.1f}% of system RAM")
    print(
        f"🏆 Safely stayed under memory limits: {'✅' if final_memory < (total_ram * 0.9) else '❌'}"
    )

    print(f"\n📈 Scaling Analysis:")
    print(f"  • Successfully ran {model_desc} ({model_params:,} params)")
    print(f"  • Fusion model: {total_params:,} total params")
    print(f"  • Training: {trainable_params:,} trainable params")
    print(f"  • Memory usage: {final_memory:.1f}GB for this scale")

    # Estimate Llama-3-8B requirements
    llama_scale_factor = 8_000_000_000 / model_params  # 8B vs current model
    estimated_llama_memory = final_memory * llama_scale_factor

    print(f"\n🔮 Llama-3-8B Estimation:")
    print(f"  • Scale factor: {llama_scale_factor:.1f}x larger than {model_desc}")
    print(f"  • Estimated memory: {estimated_llama_memory:.1f} GB")
    print(f"  • Feasible with 36GB RAM: {'✅' if estimated_llama_memory < 32 else '❌'}")

    if estimated_llama_memory < 32:
        print(f"  💡 Llama-3-8B training should be possible!")
    else:
        print(f"  💡 Would need additional optimization (quantization, etc.)")


if __name__ == "__main__":
    main()
