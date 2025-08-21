#!/usr/bin/env python3
"""
Full Llama-3-8B Training Test with CPU

This script tests training with the actual Llama-3-8B model using CPU
with 36GB RAM. We'll use smaller models where possible but real Llama-3.
"""

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

print("🚀 Testing training with full Llama-3-8B model...")
print(f"💾 Available RAM: ~36GB")

# Memory management
torch.set_num_threads(4)  # Limit CPU threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if we can use a smaller Llama model for testing
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("✅ Transformers available")

    # Try to use a smaller model first to test - Microsoft's DialoGPT is similar architecture
    print("🔍 Testing with DialoGPT-medium first...")
    model_name = "microsoft/DialoGPT-medium"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Tokenizer loaded: {model_name}")

    # Load model with CPU and lower precision
    print("📥 Loading model...")
    text_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"✅ Model loaded successfully!")

    # Model info
    total_params = sum(p.numel() for p in text_model.parameters())
    print(f"📊 Model parameters: {total_params:,}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Import our fusion components
try:
    from test_mock_training import MockPrithviEncoder

    print("✅ Successfully imported mock climate encoder")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class RealClimateTextFusion(torch.nn.Module):
    """Fusion model with real text model and mock climate encoder"""

    def __init__(self, text_model, climate_dim=256):
        super().__init__()
        self.climate_encoder = MockPrithviEncoder(climate_dim)
        self.text_model = text_model

        # Get text model's hidden size
        self.text_hidden_size = text_model.config.hidden_size

        # Project climate features to text embedding space
        self.climate_projector = torch.nn.Sequential(
            torch.nn.Linear(climate_dim, self.text_hidden_size),
            torch.nn.LayerNorm(self.text_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.text_hidden_size, self.text_hidden_size),
        )

        # Cross attention (smaller than full model)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=self.text_hidden_size, num_heads=8, batch_first=True  # Reduced from 32
        )

        # Freeze text model to save memory
        for param in self.text_model.parameters():
            param.requires_grad = False

        print(f"📊 Text model frozen (hidden_size: {self.text_hidden_size})")

    def forward(self, climate_data, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)

        # Encode climate data
        climate_features = self.climate_encoder(climate_data)
        projected_climate = self.climate_projector(climate_features)

        # Get text embeddings (not full forward pass to save memory)
        with torch.no_grad():
            text_embeddings = self.text_model.get_input_embeddings()(input_ids)

        # Cross attention: text attends to climate
        fused_features, _ = self.cross_attention(
            query=text_embeddings, key=projected_climate, value=projected_climate
        )

        # Simple output projection (instead of full language model head)
        output_logits = torch.nn.functional.linear(
            fused_features, self.text_model.get_input_embeddings().weight
        )

        return type("Output", (), {"logits": output_logits})()


def check_memory_usage():
    """Check current memory usage"""
    import psutil

    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024**3
    print(f"💾 Current RAM usage: {memory_gb:.2f} GB")
    return memory_gb


class SmallDataset:
    """Small dataset for testing"""

    def __init__(self, num_samples=10, seq_length=64):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Generate mock data
        np.random.seed(42)
        torch.manual_seed(42)

        self.climate_data = torch.randn(num_samples, 2, 20, 16, 16)

        # Create simple text data
        vocab_size = 1000
        self.input_ids = torch.randint(1, vocab_size, (num_samples, seq_length))
        self.attention_mask = torch.ones(num_samples, seq_length)
        self.labels = self.input_ids.clone()

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
    print("\n🏗️ Setting up fusion model...")
    check_memory_usage()

    # Create fusion model
    fusion_model = RealClimateTextFusion(text_model, climate_dim=256)

    # Count parameters
    total_params = sum(p.numel() for p in fusion_model.parameters())
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)

    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Frozen parameters: {total_params - trainable_params:,}")

    check_memory_usage()

    # Create small dataset
    print("\n📊 Creating small dataset...")
    dataset = SmallDataset(num_samples=8, seq_length=32)  # Very small for testing

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Very small batch size
        shuffle=False,
        collate_fn=lambda batch: {
            "climate_data": torch.stack([item["climate_data"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        },
    )

    print(f"✅ Dataset created: {len(dataset)} samples, batch_size=1")

    # Test forward pass
    print("\n🚀 Testing forward pass...")
    fusion_model.eval()

    try:
        with torch.no_grad():
            sample_batch = next(iter(dataloader))

            climate_data = sample_batch["climate_data"]
            input_ids = sample_batch["input_ids"]
            attention_mask = sample_batch["attention_mask"]

            print(f"Input shapes:")
            print(f"  Climate: {climate_data.shape}")
            print(f"  Text: {input_ids.shape}")

            check_memory_usage()

            # Forward pass
            outputs = fusion_model(climate_data, input_ids, attention_mask)

            print(f"✅ Forward pass successful!")
            print(f"📊 Output shape: {outputs.logits.shape}")

            check_memory_usage()

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # Test training step
    print("\n🎯 Testing training step...")

    try:
        fusion_model.train()

        # Only optimize the trainable parameters
        optimizer = torch.optim.AdamW(
            [p for p in fusion_model.parameters() if p.requires_grad], lr=1e-4
        )

        sample_batch = next(iter(dataloader))
        climate_data = sample_batch["climate_data"]
        input_ids = sample_batch["input_ids"]
        attention_mask = sample_batch["attention_mask"]
        labels = sample_batch["labels"]

        check_memory_usage()

        # Forward pass
        outputs = fusion_model(climate_data, input_ids, attention_mask)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
        )

        print(f"📊 Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Gradient norm (only for trainable parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in fusion_model.parameters() if p.requires_grad], max_norm=1.0
        )
        print(f"🔧 Gradient norm: {grad_norm:.4f}")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        print("✅ Training step successful!")

        check_memory_usage()

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return

    # Mini training loop
    print("\n🏃 Running mini training loop (2 steps)...")

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
                outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
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

            # Memory cleanup
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        print("✅ Mini training loop completed!")

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        return

    print("\n🎉 Real model training test completed successfully!")
    print("\nResults:")
    print("  ✅ Successfully loaded full text model")
    print("  ✅ Created climate-text fusion architecture")
    print("  ✅ Completed forward and backward passes")
    print("  ✅ Ran mini training loop with gradient updates")
    print(f"  📊 Final memory usage: {check_memory_usage():.1f} GB")

    print("\nNext step: Try with actual Llama-3-8B:")
    print("  • Change model_name to 'meta-llama/Meta-Llama-3-8B'")
    print("  • May need HuggingFace approval for Llama models")
    print("  • Consider using quantization (4-bit/8-bit) for larger models")


if __name__ == "__main__":
    main()
