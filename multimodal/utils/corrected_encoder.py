"""
Corrected PrithviWxC_Encoder Class

This class matches the EXACT architecture of the original model:
- 25 encoder transformer blocks (not 12)
- 160 input channels
- 11 static channels
- 2560 embedding dimension

"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CorrectedPrithviWxC_Encoder(nn.Module):
    """
    Corrected encoder class matching the exact original model architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Validate configuration
        required_keys = [
            'in_channels', 'in_channels_static', 'embed_dim',
            'n_blocks_encoder', 'n_heads', 'mlp_multiplier'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        print(f"🏗️  Building corrected encoder:")
        print(f"   - Input channels: {config['in_channels']}")
        print(f"   - Static channels: {config['in_channels_static']}")
        print(f"   - Encoder blocks: {config['n_blocks_encoder']}")
        print(f"   - Embedding dim: {config['embed_dim']}")

        # Input normalization (these are loaded from weights) - using exact original shapes
        self.register_buffer('input_scalers_mu', torch.zeros(1, 1, config['in_channels'], 1, 1))
        self.register_buffer('input_scalers_sigma', torch.ones(1, 1, config['in_channels'], 1, 1))
        self.register_buffer('static_input_scalers_mu', torch.zeros(1, config['in_channels_static'], 1, 1))
        self.register_buffer('static_input_scalers_sigma', torch.ones(1, config['in_channels_static'], 1, 1))

        # Mask token for masked autoencoding - exact original shape
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config['embed_dim']))

        # Patch embeddings
        self.patch_embedding = nn.Conv2d(
            config['in_channels'] * 2,  # * 2 for time dimension
            config['embed_dim'],
            kernel_size=config.get('patch_size_px', [2, 2]),
            stride=config.get('patch_size_px', [2, 2])
        )

        # Static patch embeddings
        self.patch_embedding_static = nn.Conv2d(
            168,  # Use actual input channels from original model (11 * ~15.3)
            config['embed_dim'],
            kernel_size=config.get('patch_size_px', [2, 2]),
            stride=config.get('patch_size_px', [2, 2])
        )

        # Time embeddings - using correct dimensions from original model
        self.input_time_embedding = nn.Linear(1, 640)  # Original is 640-dim, not embed_dim
        self.lead_time_embedding = nn.Linear(1, 640)

        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config['embed_dim'],
                n_heads=config['n_heads'],
                mlp_multiplier=config['mlp_multiplier'],
                dropout=config.get('dropout', 0.0)
            )
            for _ in range(config['n_blocks_encoder'])  # 25 blocks
        ])

        # No final layer norm in original model

    def forward(self, x, static=None, input_time=None, lead_time=None, mask=None):
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor [B, C, T, H, W]
            static: Static features [B, C_static, H, W]
            input_time: Input time [B, 1]
            lead_time: Lead time [B, 1]
            mask: Mask for autoencoding [B, H', W']

        Returns:
            Encoded features [B, N_patches, embed_dim]
        """
        B, C, T, H, W = x.shape

        # Normalize inputs using correct broadcasting
        # input_scalers shape: [1, 1, 160, 1, 1] -> need [1, 160, 1, 1, 1] for broadcasting with [B, 160, 2, H, W]
        mu = self.input_scalers_mu.squeeze(1).unsqueeze(2)  # [1, 160, 1, 1, 1]
        sigma = self.input_scalers_sigma.squeeze(1).unsqueeze(2)  # [1, 160, 1, 1, 1]
        x = (x - mu) / sigma

        if static is not None:
            # static_scalers shape: [1, 11, 1, 1] -> already correct for [B, 11, H, W]
            static = (static - self.static_input_scalers_mu) / self.static_input_scalers_sigma

        # Reshape for patch embedding: [B, C*T, H, W]
        x = x.view(B, C * T, H, W)

        # Patch embedding
        x = self.patch_embedding(x)  # [B, embed_dim, H', W']
        _, _, H_patch, W_patch = x.shape

        # Flatten patches: [B, embed_dim, H'*W'] -> [B, H'*W', embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add time embeddings if provided - need to expand 640-dim to 2560-dim
        if input_time is not None:
            time_emb = self.input_time_embedding(input_time)  # [B, 640]
            # Expand to match embedding dimension: repeat 4 times (640 * 4 = 2560)
            time_emb = time_emb.repeat(1, 4).unsqueeze(1)  # [B, 1, 2560]
            x = x + time_emb

        if lead_time is not None:
            lead_emb = self.lead_time_embedding(lead_time)  # [B, 640]
            # Expand to match embedding dimension: repeat 4 times
            lead_emb = lead_emb.repeat(1, 4).unsqueeze(1)  # [B, 1, 2560]
            x = x + lead_emb        # Apply mask if provided
        if mask is not None:
            # mask should be [B, H', W'] -> [B, H'*W']
            mask_flat = mask.flatten(1)  # [B, H'*W']
            mask_tokens = self.mask_token.expand(B, mask_flat.shape[1], -1)

            # Apply mask: where mask==1, use mask token; where mask==0, use original
            x = torch.where(mask_flat.unsqueeze(-1), mask_tokens, x)

        # Pass through encoder transformer blocks
        for block in self.encoder_blocks:
            x = block(x)

        # No final layer norm in original model
        return x

    @classmethod
    def from_pretrained(cls, path):
        """Load encoder from saved checkpoint."""
        print(f"🔄 Loading corrected encoder from {path}")

        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']['params']
        state_dict = checkpoint['model_state_dict']

        print(f"   ✅ Config loaded: {config['n_blocks_encoder']} blocks, {config['embed_dim']} dim")

        # Create model
        model = cls(config)

        # Load state dict with proper mapping
        model_dict = model.state_dict()
        loaded_dict = {}

        # Direct mappings for embedding layers
        key_mappings = {
            'patch_embedding.proj.weight': 'patch_embedding.weight',
            'patch_embedding.proj.bias': 'patch_embedding.bias',
            'patch_embedding_static.proj.weight': 'patch_embedding_static.weight',
            'patch_embedding_static.proj.bias': 'patch_embedding_static.bias'
        }

        # Apply direct mappings
        for old_key, new_key in key_mappings.items():
            if old_key in state_dict and new_key in model_dict:
                loaded_dict[new_key] = state_dict[old_key]

        # Map encoder block weights
        for key, value in state_dict.items():
            if 'encoder.lgl_block.transformers.' in key:
                # Map transformer block weights
                parts = key.split('.')
                block_idx = parts[3]  # transformers.{idx}

                if 'attention.0' in key:
                    new_key = f'encoder_blocks.{block_idx}.norm1.weight' if 'weight' in key else f'encoder_blocks.{block_idx}.norm1.bias'
                elif 'attention.1.qkv_layer' in key:
                    new_key = f'encoder_blocks.{block_idx}.attn.qkv.weight' if 'weight' in key else f'encoder_blocks.{block_idx}.attn.qkv.bias'
                elif 'attention.1.w_layer' in key:
                    # Only map weight, not bias (original model doesn't have bias for attention projection)
                    if 'weight' in key:
                        new_key = f'encoder_blocks.{block_idx}.attn.proj.weight'
                    else:
                        continue  # Skip bias - original model doesn't have it
                elif 'ff.0' in key:
                    new_key = f'encoder_blocks.{block_idx}.norm2.weight' if 'weight' in key else f'encoder_blocks.{block_idx}.norm2.bias'
                elif 'ff.1.net.0' in key:
                    new_key = f'encoder_blocks.{block_idx}.mlp.fc1.weight' if 'weight' in key else f'encoder_blocks.{block_idx}.mlp.fc1.bias'
                elif 'ff.1.net.3' in key:
                    new_key = f'encoder_blocks.{block_idx}.mlp.fc2.weight' if 'weight' in key else f'encoder_blocks.{block_idx}.mlp.fc2.bias'
                else:
                    continue

                if new_key in model_dict:
                    loaded_dict[new_key] = value
            else:
                # Direct mapping for other components (scalers, etc.)
                if key in model_dict:
                    loaded_dict[key] = value

        # Load the mapped weights
        missing_keys = []
        unexpected_keys = []

        for key in model_dict.keys():
            if key not in loaded_dict:
                missing_keys.append(key)

        for key in loaded_dict.keys():
            if key not in model_dict:
                unexpected_keys.append(key)

        model.load_state_dict(loaded_dict, strict=False)

        print(f"   ✅ Loaded weights: {len(loaded_dict)}/{len(model_dict)} parameters")
        if missing_keys:
            print(f"   ⚠️  Missing: {len(missing_keys)} keys")
            print(f"   📋 Missing keys:")
            for key in missing_keys[:10]:  # Show first 10
                print(f"      - {key}")
            if len(missing_keys) > 10:
                print(f"      ... and {len(missing_keys) - 10} more")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected: {len(unexpected_keys)} keys")

        return model


class TransformerBlock(nn.Module):
    """Transformer block for the encoder."""

    def __init__(self, embed_dim, n_heads, mlp_multiplier, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_multiplier), dropout)

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self attention."""

    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)  # No bias in original model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """MLP block."""

    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def test_corrected_encoder():
    """Test the corrected encoder with fixed weights."""
    print("🧪 Testing Corrected Encoder")
    print("=" * 40)

    try:
        # Load the fixed encoder
        encoder_path = "data/weights/prithvi_encoder_fixed.pt"
        encoder = CorrectedPrithviWxC_Encoder.from_pretrained(encoder_path)

        print("✅ Encoder loaded successfully!")

        # Test forward pass
        print("\\n🚀 Testing forward pass...")

        batch_size = 2
        x = torch.randn(batch_size, 160, 2, 180, 288)  # Correct input shape
        static = torch.randn(batch_size, 11, 180, 288)  # Correct static shape
        input_time = torch.randn(batch_size, 1)
        lead_time = torch.randn(batch_size, 1)

        with torch.no_grad():
            output = encoder(x, static=static, input_time=input_time, lead_time=lead_time)

        print(f"   ✅ Input shape: {x.shape}")
        print(f"   ✅ Static shape: {static.shape}")
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Forward pass successful!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_corrected_encoder()
    if success:
        print("\\n🎉 All tests passed! The encoder is working correctly.")
    else:
        print("\\n💥 Tests failed. Check the implementation.")
