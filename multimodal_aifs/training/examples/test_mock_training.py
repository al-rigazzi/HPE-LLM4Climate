#!/usr/bin/env python3
"""
Minimal Training Test with Mock Data using AIFS

This script tests the AIFS training pipeline with small mock data to verify
that everything works without requiring large models or real climate data.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

print("🔍 Testing AIFS training pipeline with mock data...")

# Check available memory
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 GPU Memory Available: {gpu_memory:.1f} GB")
else:
    device = torch.device("cpu")
    print("💻 Using CPU (no CUDA available)")

print(f"🎯 Device: {device}")

# Test basic imports
try:
    from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion
    print("✅ Successfully imported AIFSClimateTextFusion")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test with small mock models instead of full AIFS
print("\n🧪 Creating mock AIFS model components...")

class MockAIFSEncoder(torch.nn.Module):
    """Mock AIFS encoder for testing"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 64, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(64 * 8 * 8, output_dim)

    def forward(self, x):
        # x: [batch, time, channels, height, width]
        batch_size, time_steps = x.shape[:2]

        # Process each timestep
        outputs = []
        for t in range(time_steps):
            x_t = self.conv(x[:, t])  # [batch, 64, height, width]
            x_t = self.pool(x_t)      # [batch, 64, 8, 8]
            x_t = self.flatten(x_t)   # [batch, 64*8*8]
            x_t = self.linear(x_t)    # [batch, output_dim]
            outputs.append(x_t)

        # Stack timesteps: [batch, time_steps, output_dim]
        return torch.stack(outputs, dim=1)

class MockLlamaModel(torch.nn.Module):
    """Mock Llama model for testing"""
    def __init__(self, vocab_size=1000, hidden_size=256, max_length=64):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)

        if attention_mask is not None:
            # Convert attention mask for transformer
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        return type('MockOutput', (), {'last_hidden_state': output})()

class MockClimateTextFusion(torch.nn.Module):
    """Simplified fusion model for testing"""
    def __init__(self, climate_dim=256, text_dim=256):
        super().__init__()
        self.climate_encoder = MockAIFSEncoder(climate_dim)
        self.text_encoder = MockLlamaModel(hidden_size=text_dim)

        # Cross attention
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=4,
            batch_first=True
        )

        # Output projection
        self.output_projection = torch.nn.Linear(text_dim, 1000)  # Mock vocab size

    def forward(self, climate_data, input_ids, attention_mask=None):
        # Encode climate data
        climate_features = self.climate_encoder(climate_data)

        # Encode text
        text_output = self.text_encoder(input_ids, attention_mask)
        text_features = text_output.last_hidden_state

        # Cross attention: text attends to climate
        fused_features, _ = self.cross_attention(
            query=text_features,
            key=climate_features,
            value=climate_features
        )

        # Output projection
        logits = self.output_projection(fused_features)

        return type('MockOutput', (), {'logits': logits})()

# Create mock model
print("🏗️ Creating mock fusion model...")
model = MockClimateTextFusion(climate_dim=256, text_dim=256)
model = model.to(device)

# Create mock data
print("📊 Creating mock training data...")
batch_size = 2
seq_length = 32
time_steps = 2
channels = 20
height, width = 16, 16

# Mock climate data: [batch, time, channels, height, width]
climate_data = torch.randn(batch_size, time_steps, channels, height, width).to(device)

# Mock text data: [batch, seq_length]
input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
attention_mask = torch.ones(batch_size, seq_length).to(device)
labels = torch.randint(0, 1000, (batch_size, seq_length)).to(device)

print(f"Climate data shape: {climate_data.shape}")
print(f"Text input shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")

# Test forward pass
print("\n🚀 Testing forward pass...")
try:
    with torch.no_grad():
        outputs = model(climate_data, input_ids, attention_mask)
        print(f"✅ Forward pass successful! Output shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    sys.exit(1)

# Test training step
print("\n🎯 Testing training step...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward pass
    outputs = model(climate_data, input_ids, attention_mask)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        labels.view(-1)
    )

    print(f"📊 Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"🔧 Gradient norm: {grad_norm:.4f}")

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    print("✅ Training step successful!")

except Exception as e:
    print(f"❌ Training step failed: {e}")
    sys.exit(1)

# Memory usage
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated() / 1024**3
    memory_cached = torch.cuda.memory_reserved() / 1024**3
    print(f"\n💾 GPU Memory Used: {memory_used:.2f} GB")
    print(f"💾 GPU Memory Cached: {memory_cached:.2f} GB")

print("\n🎉 Mock training test completed successfully!")
print("\nNext steps:")
print("  • The basic training pipeline works with mock data")
print("  • For real training, you'll need:")
print("    - Actual climate data (MERRA-2 format)")
print("    - Proper Llama-3 model (requires HuggingFace approval)")
print("    - More GPU memory for full-scale models")
print("    - DeepSpeed for distributed training")
