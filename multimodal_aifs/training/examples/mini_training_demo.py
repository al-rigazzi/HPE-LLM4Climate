#!/usr/bin/env python3
"""
Mini Training Loop with Mock Data

This script demonstrates a complete training loop with mock data,
including validation, checkpointing, and metrics tracking.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

print("🚀 Starting mini training demonstration...")

# Use the mock model from the previous test
from test_mock_training import MockClimateTextFusion

class MockDataset:
    """Mock dataset for demonstration"""
    def __init__(self, num_samples=20, seq_length=32):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Generate consistent mock data
        np.random.seed(42)
        torch.manual_seed(42)

        self.climate_data = torch.randn(num_samples, 2, 20, 16, 16)
        self.input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        self.attention_mask = torch.ones(num_samples, seq_length)
        self.labels = torch.randint(0, 1000, (num_samples, seq_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'climate_data': self.climate_data[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def create_dataloader(dataset, batch_size=2, shuffle=True):
    """Create dataloader with collate function"""
    def collate_fn(batch):
        return {
            'climate_data': torch.stack([item['climate_data'] for item in batch]),
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move data to device
        climate_data = batch['climate_data'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(climate_data, input_ids, attention_mask)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
            'grad_norm': f'{grad_norm:.3f}'
        })

    return total_loss / num_batches

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            climate_data = batch['climate_data'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(climate_data, input_ids, attention_mask)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def main():
    # Training configuration
    config = {
        'epochs': 3,
        'batch_size': 2,
        'learning_rate': 1e-4,
        'train_samples': 20,
        'val_samples': 8,
        'save_dir': 'mock_checkpoints'
    }

    print(f"📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Device: {device}")

    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = MockDataset(num_samples=config['train_samples'])
    val_dataset = MockDataset(num_samples=config['val_samples'])

    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=config['batch_size'])
    val_loader = create_dataloader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")

    # Create model
    print("\n🏗️ Creating model...")
    model = MockClimateTextFusion(climate_dim=256, text_dim=256)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")

    train_losses = []
    val_losses = []

    for epoch in range(1, config['epochs'] + 1):
        print(f"\n📈 Epoch {epoch}/{config['epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Print epoch summary
        print(f"📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")

        # Memory usage (if GPU)
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU Memory: {memory_used:.2f} GB")

        # Save checkpoint
        os.makedirs(config['save_dir'], exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, f"{config['save_dir']}/checkpoint_epoch_{epoch}.pt")

    # Final results
    print(f"\n🎉 Training completed!")
    print(f"📈 Final train loss: {train_losses[-1]:.4f}")
    print(f"📉 Final val loss: {val_losses[-1]:.4f}")

    # Save training history
    history = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    with open(f"{config['save_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"💾 Results saved to {config['save_dir']}/")

    # Test inference
    print(f"\n🧪 Testing inference...")
    model.eval()

    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(val_loader))
        climate_data = sample_batch['climate_data'].to(device)
        input_ids = sample_batch['input_ids'].to(device)
        attention_mask = sample_batch['attention_mask'].to(device)

        # Run inference
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if torch.cuda.is_available():
            start_time.record()

        outputs = model(climate_data, input_ids, attention_mask)

        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            print(f"⚡ Inference time: {inference_time:.2f} ms")

        print(f"✅ Inference successful! Output shape: {outputs.logits.shape}")

if __name__ == "__main__":
    main()
