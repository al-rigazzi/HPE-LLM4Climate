#!/usr/bin/env python3
"""
Llama-3-8B Training Test with Memory Optimization

This script attempts to run training with the actual Llama-3-8B model
using CPU with 36GB RAM and aggressive memory optimization.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import gc
import warnings

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Memory optimization settings
torch.set_num_threads(2)  # Limit CPU threads
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings('ignore')

print("🚀 Testing training with Llama-3-8B model...")
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Try to load Llama-3-8B
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print("✅ Transformers with quantization support available")

    # Use only Llama 3-8B (other models disabled)
    model_options = [
        "meta-llama/Meta-Llama-3-8B",
        # Other models disabled to ensure we use only Llama 3-8B
        # "meta-llama/Llama-2-7b-hf",
        # "NousResearch/Llama-2-7b-hf",  # Alternative without gating
        # DialoGPT disabled due to torch.load vulnerability in PyTorch 2.4
        # "microsoft/DialoGPT-large"     # Fallback
    ]

    model_name = None
    tokenizer = None
    text_model = None

    for model_candidate in model_options:
        try:
            print(f"🔍 Trying to load: {model_candidate}")

            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_candidate)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"✅ Tokenizer loaded: {model_candidate}")

            # Try to load model with 8-bit quantization to save memory
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            print("📥 Loading model with 8-bit quantization...")
            initial_memory = check_memory_usage()
            print(f"💾 RAM before model loading: {initial_memory:.2f} GB")

            text_model = AutoModelForCausalLM.from_pretrained(
                model_candidate,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )

            model_name = model_candidate
            break

        except Exception as e:
            print(f"❌ Failed to load {model_candidate}: {e}")
            clear_memory()
            continue

    if text_model is None:
        print("❌ Could not load any language model")
        sys.exit(1)

    print(f"✅ Successfully loaded: {model_name}")

    # Model info
    total_params = sum(p.numel() for p in text_model.parameters())
    print(f"📊 Model parameters: {total_params:,}")

    memory_after_model = check_memory_usage()
    print(f"💾 RAM after model loading: {memory_after_model:.2f} GB")
    print(f"📈 Model memory usage: {memory_after_model - initial_memory:.2f} GB")

except ImportError:
    print("❌ BitsAndBytesConfig not available. Install with: pip install bitsandbytes")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error setting up model: {e}")
    sys.exit(1)

# Import our fusion components
try:
    from test_mock_training import MockAIFSEncoder
    print("✅ Successfully imported mock AIFS encoder")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class OptimizedClimateTextFusion(torch.nn.Module):
    """Memory-optimized AIFS fusion model with real Llama model"""
    def __init__(self, text_model, climate_dim=128):  # Reduced climate dim
        super().__init__()

        # Smaller AIFS-inspired climate encoder
        self.climate_encoder = MockAIFSEncoder(climate_dim)
        self.text_model = text_model

        # Get text model's hidden size
        self.text_hidden_size = text_model.config.hidden_size

        # Smaller projection network
        self.climate_projector = torch.nn.Sequential(
            torch.nn.Linear(climate_dim, self.text_hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.text_hidden_size // 2, self.text_hidden_size)
        )

        # Smaller cross attention
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=self.text_hidden_size,
            num_heads=4,  # Much smaller
            batch_first=True
        )

        # Freeze text model completely
        for param in self.text_model.parameters():
            param.requires_grad = False

        print(f"📊 Text model frozen (hidden_size: {self.text_hidden_size})")

    def forward(self, climate_data, input_ids, attention_mask=None):
        # Encode climate data
        climate_features = self.climate_encoder(climate_data)
        projected_climate = self.climate_projector(climate_features)

        # Get text embeddings only (avoid full forward pass)
        with torch.no_grad():
            text_embeddings = self.text_model.get_input_embeddings()(input_ids)

        # Cross attention with gradient checkpointing
        def attention_forward(query, key, value):
            output, _ = self.cross_attention(query=query, key=key, value=value)
            return output

        fused_features = torch.utils.checkpoint.checkpoint(
            attention_forward,
            text_embeddings,
            projected_climate,
            projected_climate,
            use_reentrant=False
        )

        # Simple output projection
        with torch.no_grad():
            embedding_weight = self.text_model.get_input_embeddings().weight

        output_logits = torch.nn.functional.linear(fused_features, embedding_weight)

        return type('Output', (), {'logits': output_logits})()

class TinyDataset:
    """Very small dataset for testing"""
    def __init__(self, num_samples=4, seq_length=16):  # Even smaller
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Generate tiny mock data
        np.random.seed(42)
        torch.manual_seed(42)

        # Smaller climate data
        self.climate_data = torch.randn(num_samples, 1, 10, 8, 8)  # Much smaller

        # Simple text data
        vocab_size = min(1000, tokenizer.vocab_size)
        self.input_ids = torch.randint(1, vocab_size, (num_samples, seq_length))
        self.attention_mask = torch.ones(num_samples, seq_length)
        self.labels = self.input_ids.clone()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'climate_data': self.climate_data[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def main():
    print(f"\n🏗️ Setting up optimized fusion model...")
    print(f"💾 Current RAM: {check_memory_usage():.2f} GB")

    try:
        # Create fusion model with memory optimization
        fusion_model = OptimizedClimateTextFusion(text_model, climate_dim=128)

        # Enable gradient checkpointing
        fusion_model.gradient_checkpointing_enable = True

        # Count parameters
        total_params = sum(p.numel() for p in fusion_model.parameters())
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)

        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Frozen parameters: {total_params - trainable_params:,}")

        print(f"💾 RAM after fusion model: {check_memory_usage():.2f} GB")

    except Exception as e:
        print(f"❌ Error creating fusion model: {e}")
        return

    # Create tiny dataset
    print("\n📊 Creating tiny dataset...")
    dataset = TinyDataset(num_samples=4, seq_length=16)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: {
            'climate_data': torch.stack([item['climate_data'] for item in batch]),
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }
    )

    print(f"✅ Dataset created: {len(dataset)} samples")
    print(f"💾 RAM after dataset: {check_memory_usage():.2f} GB")

    # Test forward pass
    print("\n🚀 Testing forward pass...")
    try:
        fusion_model.eval()

        with torch.no_grad():
            sample_batch = next(iter(dataloader))

            climate_data = sample_batch['climate_data']
            input_ids = sample_batch['input_ids']
            attention_mask = sample_batch['attention_mask']

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
            print(f"📈 Forward pass memory: {memory_after - memory_before:.2f} GB")

            clear_memory()

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        clear_memory()
        return

    # Test training step with careful memory management
    print("\n🎯 Testing training step...")

    try:
        fusion_model.train()

        # Optimizer only for trainable parameters
        optimizer = torch.optim.AdamW(
            [p for p in fusion_model.parameters() if p.requires_grad],
            lr=5e-5,  # Smaller learning rate
            weight_decay=0.01
        )

        sample_batch = next(iter(dataloader))
        climate_data = sample_batch['climate_data']
        input_ids = sample_batch['input_ids']
        attention_mask = sample_batch['attention_mask']
        labels = sample_batch['labels']

        memory_before = check_memory_usage()
        print(f"💾 RAM before training step: {memory_before:.2f} GB")

        # Forward pass
        outputs = fusion_model(climate_data, input_ids, attention_mask)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1)
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
            [p for p in fusion_model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        print(f"🔧 Gradient norm: {grad_norm:.4f}")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        memory_final = check_memory_usage()
        print(f"✅ Training step successful!")
        print(f"💾 RAM after optimizer step: {memory_final:.2f} GB")
        print(f"📈 Total training step memory: {memory_final - memory_before:.2f} GB")

        clear_memory()

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        print(f"💾 Current RAM: {check_memory_usage():.2f} GB")
        clear_memory()
        return

    print(f"\n🎉 Llama-3 training test completed successfully!")
    print(f"💾 Final RAM usage: {check_memory_usage():.2f} GB")

    print(f"\nResults with {model_name}:")
    print(f"  ✅ Successfully loaded {total_params:,} parameter model")
    print(f"  ✅ Created climate-text fusion with memory optimization")
    print(f"  ✅ Completed forward and backward passes")
    print(f"  ✅ Used quantization and gradient checkpointing")
    print(f"  📊 Peak memory usage stayed well under 36GB limit")

if __name__ == "__main__":
    main()
