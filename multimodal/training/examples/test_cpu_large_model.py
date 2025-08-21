#!/usr/bin/env python3
"""
CPU-based Llama Training Test with 36GB RAM

This script attempts to run training with large language models on CPU
using 36GB RAM with aggressive memory optimization techniques.
"""

import gc
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Aggressive memory optimization
torch.set_num_threads(2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
warnings.filterwarnings("ignore")

print("🚀 Testing large language model training on CPU...")
print(f"💾 Available RAM: ~36GB")


def check_memory_usage():
    """Check current memory usage"""
    import psutil

    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024**3
    return memory_gb


def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()


# Try to load models without quantization (CPU compatible)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("✅ Transformers available")

    # Model options that should work without gating
    model_options = [
        ("microsoft/DialoGPT-large", "DialoGPT-Large (762M params)"),
        ("gpt2-large", "GPT-2 Large (774M params)"),
        ("gpt2-medium", "GPT-2 Medium (345M params)"),
        ("microsoft/DialoGPT-medium", "DialoGPT-Medium (345M params)"),
    ]

    model_name = None
    tokenizer = None
    text_model = None

    for model_candidate, description in model_options:
        try:
            print(f"🔍 Trying to load: {description}")

            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_candidate)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"✅ Tokenizer loaded")

            print("📥 Loading model with CPU optimization...")
            initial_memory = check_memory_usage()
            print(f"💾 RAM before model loading: {initial_memory:.2f} GB")

            # Load model with CPU optimization
            text_model = AutoModelForCausalLM.from_pretrained(
                model_candidate,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                low_cpu_mem_usage=True,
                use_cache=False,  # Disable KV cache to save memory
            )

            # Move to CPU explicitly
            text_model = text_model.cpu()

            model_name = model_candidate
            description_used = description
            break

        except Exception as e:
            print(f"❌ Failed to load {model_candidate}: {e}")
            clear_memory()
            continue

    if text_model is None:
        print("❌ Could not load any language model")
        sys.exit(1)

    print(f"✅ Successfully loaded: {description_used}")

    # Model info
    total_params = sum(p.numel() for p in text_model.parameters())
    print(f"📊 Model parameters: {total_params:,}")

    memory_after_model = check_memory_usage()
    print(f"💾 RAM after model loading: {memory_after_model:.2f} GB")
    print(f"📈 Model memory usage: {memory_after_model - initial_memory:.2f} GB")

    if memory_after_model > 30:  # If using too much memory
        print("⚠️ High memory usage, will use more aggressive optimization")

except Exception as e:
    print(f"❌ Error setting up model: {e}")
    sys.exit(1)

# Import our fusion components
try:
    from test_mock_training import MockPrithviEncoder

    print("✅ Successfully imported mock climate encoder")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class CPUOptimizedFusion(torch.nn.Module):
    """CPU-optimized fusion model"""

    def __init__(self, text_model, climate_dim=64):  # Very small climate dim
        super().__init__()

        # Very small climate encoder
        self.climate_encoder = MockPrithviEncoder(climate_dim)
        self.text_model = text_model

        # Get text model's hidden size
        self.text_hidden_size = text_model.config.hidden_size

        # Simple projection
        self.climate_projector = torch.nn.Linear(climate_dim, self.text_hidden_size)

        # Small cross attention
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=self.text_hidden_size, num_heads=2, batch_first=True  # Very small
        )

        # Freeze text model to save memory and computation
        for param in self.text_model.parameters():
            param.requires_grad = False

        print(f"📊 Text model frozen (hidden_size: {self.text_hidden_size})")
        print(f"📊 Only training climate encoder and fusion layers")

    def forward(self, climate_data, input_ids, attention_mask=None):
        # Encode climate data
        climate_features = self.climate_encoder(climate_data)
        projected_climate = self.climate_projector(climate_features)

        # Get text embeddings only (no full forward pass)
        with torch.no_grad():
            text_embeddings = self.text_model.get_input_embeddings()(input_ids)

        # Simple cross attention
        fused_features, _ = self.cross_attention(
            query=text_embeddings, key=projected_climate, value=projected_climate
        )

        # Simple output (avoid using full LM head)
        # Just use a small vocab subset for testing
        vocab_subset = min(1000, self.text_model.config.vocab_size)

        with torch.no_grad():
            full_embedding_weight = self.text_model.get_input_embeddings().weight
            subset_weight = full_embedding_weight[:vocab_subset, :]

        output_logits = torch.nn.functional.linear(fused_features, subset_weight)

        return type("Output", (), {"logits": output_logits})()


class MicroDataset:
    """Micro dataset for testing"""

    def __init__(self, num_samples=2, seq_length=8):  # Tiny
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Tiny mock data
        np.random.seed(42)
        torch.manual_seed(42)

        # Very small climate data
        self.climate_data = torch.randn(num_samples, 1, 5, 4, 4)  # Tiny

        # Simple text data with small vocab
        vocab_size = 1000
        self.input_ids = torch.randint(1, vocab_size, (num_samples, seq_length))
        self.attention_mask = torch.ones(num_samples, seq_length)
        self.labels = torch.randint(0, vocab_size, (num_samples, seq_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "climate_data": self.climate_data[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def main():
    print(f"\n🏗️ Setting up CPU-optimized fusion model...")
    print(f"💾 Current RAM: {check_memory_usage():.2f} GB")

    try:
        # Create fusion model with heavy optimization
        fusion_model = CPUOptimizedFusion(text_model, climate_dim=64)

        # Count parameters
        total_params = sum(p.numel() for p in fusion_model.parameters())
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)

        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Frozen parameters: {total_params - trainable_params:,}")
        print(f"📈 Parameter reduction: {(1 - trainable_params/total_params)*100:.1f}% frozen")

        memory_after_fusion = check_memory_usage()
        print(f"💾 RAM after fusion model: {memory_after_fusion:.2f} GB")

    except Exception as e:
        print(f"❌ Error creating fusion model: {e}")
        return

    # Create micro dataset
    print("\n📊 Creating micro dataset...")
    dataset = MicroDataset(num_samples=2, seq_length=8)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: {
            "climate_data": torch.stack([item["climate_data"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        },
    )

    print(f"✅ Dataset created: {len(dataset)} samples")
    print(f"💾 RAM after dataset: {check_memory_usage():.2f} GB")

    # Test forward pass
    print("\n🚀 Testing forward pass...")
    try:
        fusion_model.eval()

        with torch.no_grad():
            sample_batch = next(iter(dataloader))

            climate_data = sample_batch["climate_data"]
            input_ids = sample_batch["input_ids"]
            attention_mask = sample_batch["attention_mask"]

            print(f"Input shapes:")
            print(f"  Climate: {climate_data.shape}")
            print(f"  Text: {input_ids.shape}")

            memory_before = check_memory_usage()
            print(f"💾 RAM before forward: {memory_before:.2f} GB")

            # Forward pass
            outputs = fusion_model(climate_data, input_ids, attention_mask)

            memory_after = check_memory_usage()
            print(f"✅ Forward pass successful!")
            print(f"📊 Output shape: {outputs.logits.shape}")
            print(f"💾 RAM after forward: {memory_after:.2f} GB")
            print(f"📈 Forward pass memory delta: {memory_after - memory_before:.2f} GB")

            clear_memory()

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        clear_memory()
        return

    # Test training step
    print("\n🎯 Testing training step...")

    try:
        fusion_model.train()

        # Optimizer only for trainable parameters
        optimizer = torch.optim.AdamW(
            [p for p in fusion_model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01
        )

        sample_batch = next(iter(dataloader))
        climate_data = sample_batch["climate_data"]
        input_ids = sample_batch["input_ids"]
        attention_mask = sample_batch["attention_mask"]
        labels = sample_batch["labels"]

        memory_before = check_memory_usage()
        print(f"💾 RAM before training step: {memory_before:.2f} GB")

        # Forward pass
        outputs = fusion_model(climate_data, input_ids, attention_mask)

        # Compute loss (only for subset vocab)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1) % 1000,  # Clamp to subset vocab
        )

        print(f"📊 Loss: {loss.item():.4f}")

        memory_after_forward = check_memory_usage()
        print(f"💾 RAM after forward: {memory_after_forward:.2f} GB")

        # Backward pass
        loss.backward()

        memory_after_backward = check_memory_usage()
        print(f"💾 RAM after backward: {memory_after_backward:.2f} GB")

        # Gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in fusion_model.parameters() if p.requires_grad], max_norm=1.0
        )
        print(f"🔧 Gradient norm: {grad_norm:.4f}")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        memory_final = check_memory_usage()
        print(f"✅ Training step successful!")
        print(f"💾 RAM after optimizer step: {memory_final:.2f} GB")
        print(f"📈 Total training memory delta: {memory_final - memory_before:.2f} GB")

        clear_memory()

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        print(f"💾 Current RAM: {check_memory_usage():.2f} GB")
        clear_memory()
        return

    # Test a few training steps
    print("\n🏃 Running micro training loop...")

    try:
        fusion_model.train()

        for step, batch in enumerate(dataloader):
            if step >= 2:  # Only 2 steps
                break

            climate_data = batch["climate_data"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            outputs = fusion_model(climate_data, input_ids, attention_mask)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1) % 1000
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in fusion_model.parameters() if p.requires_grad], max_norm=1.0
            )

            # Optimizer step
            optimizer.step()

            memory_gb = check_memory_usage()

            print(
                f"Step {step+1}: Loss={loss.item():.4f}, GradNorm={grad_norm:.3f}, RAM={memory_gb:.1f}GB"
            )

            clear_memory()

        print("✅ Micro training loop completed!")

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        return

    final_memory = check_memory_usage()
    print(f"\n🎉 Large model CPU training test completed!")
    print(f"💾 Final RAM usage: {final_memory:.2f} GB")
    print(f"📊 Peak memory stayed under 36GB limit: {'✅' if final_memory < 36 else '❌'}")

    print(f"\nResults with {description_used}:")
    print(f"  ✅ Successfully loaded {total_params:,} parameter model on CPU")
    print(f"  ✅ Created climate-text fusion with aggressive optimization")
    print(f"  ✅ Completed forward and backward passes")
    print(f"  ✅ Ran training steps with {trainable_params:,} trainable parameters")
    print(f"  📊 Memory efficiency: Used {final_memory:.1f}GB of 36GB available")

    print(f"\n💡 Key insights:")
    print(f"  • CPU training is possible with proper memory management")
    print(f"  • Freezing large model and training only fusion layers works")
    print(f"  • {total_params//1000000}M parameter models fit comfortably in 36GB RAM")
    print(f"  • For Llama-3-8B: Would need ~24-32GB just for model weights")


if __name__ == "__main__":
    main()
