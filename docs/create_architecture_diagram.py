#!/usr/bin/env python3
"""
Multimodal Climate-Text Fusion Model Architecture Diagram

This script creates a comprehensive diagram of the HPE LLM4Climate
multimodal fusion architecture showing:
- Data inputs and preprocessing
- Model components (PrithviWxC encoder, Llama LLM)
- Tokenization and embedding methods
- Cross-attention fusion mechanism
- Output generation

Generates both PNG and PDF versions for presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure with professional styling
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme for professional presentation
colors = {
    'climate': '#2E8B57',      # Sea Green
    'text': '#4169E1',         # Royal Blue
    'fusion': '#FF6347',       # Tomato
    'output': '#9370DB',       # Medium Purple
    'processing': '#FFD700',   # Gold
    'background': '#F8F8FF'    # Ghost White
}

# Helper function to create professional boxes
def create_box(ax, xy, width, height, text, color, text_color='white', fontsize=10, fontweight='bold'):
    """Create a professional rounded rectangle box with text"""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)

    # Add text in center of box
    text_x = xy[0] + width/2
    text_y = xy[1] + height/2
    ax.text(text_x, text_y, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color, wrap=True)

    return box

# Helper function to create arrows
def create_arrow(ax, start, end, color='black', style='->', linewidth=2):
    """Create professional arrows between components"""
    arrow = ConnectionPatch(start, end, "data", "data",
                          arrowstyle=style, shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc=color, ec=color, linewidth=linewidth)
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(8, 11.5, 'HPE LLM4Climate: Multimodal Climate-Text Fusion Architecture',
        ha='center', va='center', fontsize=18, fontweight='bold', color='black')

# =================== INPUT LAYER ===================
ax.text(8, 10.5, 'Input Layer', ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# Climate Data Input
create_box(ax, (0.5, 9), 3, 1,
           'Climate Data\n(MERRA-2)\n[B, T, C, H, W]\n[4, 73, 20, 721, 1440]',
           colors['climate'], fontsize=9)

# Text Input
create_box(ax, (12.5, 9), 3, 1,
           'Text Query\n"Climate in California"\n[B, max_length]\n[4, 512 tokens]',
           colors['text'], fontsize=9)

# =================== PREPROCESSING LAYER ===================
ax.text(8, 8.2, 'Preprocessing Layer', ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# Climate Preprocessing Chain
create_box(ax, (0.5, 7), 1.4, 0.8,
           'Normalization\nμ/σ scaling\nε=1e-6',
           colors['processing'], 'black', fontsize=8)

create_box(ax, (2.1, 7), 1.4, 0.8,
           'Patch Embedding\n16×16 patches\n→ 768 dim',
           colors['processing'], 'black', fontsize=8)

# Location Processing
create_box(ax, (4, 6), 2.5, 1.2,
           'Location Processing\n• Geographic Resolver\n• Spatial Cropping\n• Region Masking\n[lat: -90→90, lon: -180→180]',
           colors['processing'], 'black', fontsize=8)

# Text Preprocessing
create_box(ax, (12.5, 7), 3, 0.8,
           'Text Tokenization\nLlama-3 Tokenizer\n→ [4, 512] token_ids',
           colors['processing'], 'black', fontsize=8)

# =================== ENCODER LAYER ===================
ax.text(8, 5.5, 'Encoder Layer', ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# PrithviWxC Encoder
create_box(ax, (0.5, 3.5), 3, 1.5,
           'PrithviWxC Encoder\n• Patch Embed: 768 dim\n• Position Encoding\n• 12 Transformer Blocks\n• Output: [B, 1024, 768]',
           colors['climate'], fontsize=9)

# Llama Encoder
create_box(ax, (12.5, 3.5), 3, 1.5,
           'Llama-3-8B Encoder\n• Token Embeddings\n• 32 Transformer Layers\n• 4096 hidden dim\n• Output: [B, 512, 4096]',
           colors['text'], fontsize=9)

# =================== PROJECTION LAYER ===================
ax.text(8, 2.8, 'Projection Layer', ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# Climate Feature Projector
create_box(ax, (4.5, 2), 3, 0.6,
           'Climate Projector\n768 → 4096 dim\nLayerNorm + GELU',
           colors['fusion'], fontsize=9)

# =================== FUSION LAYER ===================
ax.text(8, 1.2, 'Cross-Attention Fusion', ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# Cross Attention Module
create_box(ax, (6, 0.2), 4, 0.8,
           'MultiHead Cross-Attention\n• Query: Text embeddings [B, 512, 4096]\n• Key/Value: Climate features [B, 1024, 4096]\n• 32 attention heads\n• Output: Fused features [B, 512, 4096]',
           colors['fusion'], fontsize=8)

# =================== ARROWS ===================
# Climate path arrows
create_arrow(ax, (2, 9), (1.2, 7.8), colors['climate'])
create_arrow(ax, (2.8, 7.8), (2.8, 7.8), colors['climate'])
create_arrow(ax, (2, 7), (2, 5), colors['climate'])

# Location processing arrows
create_arrow(ax, (3.5, 7.4), (4, 6.6), colors['processing'])

# Text path arrows
create_arrow(ax, (14, 9), (14, 7.8), colors['text'])
create_arrow(ax, (14, 7), (14, 5), colors['text'])

# Projection arrows
create_arrow(ax, (2, 3.5), (4.5, 2.6), colors['climate'])
create_arrow(ax, (14, 3.5), (7.5, 2.6), colors['text'])

# Fusion arrows
create_arrow(ax, (6, 2), (6.5, 1), colors['fusion'])
create_arrow(ax, (7.5, 2), (7.5, 1), colors['fusion'])

# =================== TECHNICAL SPECIFICATIONS ===================
# Add technical specs box
specs_text = """Technical Specifications:
• Climate Input: MERRA-2 reanalysis data (73 variables, 20 pressure levels)
• Spatial Resolution: 0.25° × 0.3125° (721×1440 grid points)
• Temporal Resolution: Hourly data, 4 timesteps per sample
• Text Model: Meta-Llama-3-8B (requires HuggingFace approval)
• Patch Size: 16×16 pixels for climate tokenization
• Embedding Dimension: 4096 (aligned to Llama-3)
• Attention Heads: 32 heads in cross-attention
• Location Encoding: Geographic coordinates + spatial masking
• Training: DeepSpeed with ZeRO-2, FP16 mixed precision"""

create_box(ax, (0.5, 0.2), 5, 2.5, specs_text, colors['background'], 'black', fontsize=7, fontweight='normal')

# =================== LEGEND ===================
legend_elements = [
    patches.Patch(color=colors['climate'], label='Climate Processing'),
    patches.Patch(color=colors['text'], label='Text Processing'),
    patches.Patch(color=colors['processing'], label='Preprocessing'),
    patches.Patch(color=colors['fusion'], label='Fusion Layer'),
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=10)

# =================== ANNOTATIONS ===================
# Add dimension annotations
ax.annotate('B=Batch Size, T=Time, C=Channels\nH=Height, W=Width',
            xy=(2, 8.5), xytext=(4.5, 8.5),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
            fontsize=8, color='gray')

ax.annotate('Location-aware\nSpatial Attention',
            xy=(5.2, 6.6), xytext=(7.5, 7.5),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
            fontsize=8, color='gray')

# Save the diagram
plt.tight_layout()
plt.savefig('/Users/arigazzi/Documents/DeepLearning/LLM for climate/HPE-LLM4Climate/docs/multimodal_architecture_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/arigazzi/Documents/DeepLearning/LLM for climate/HPE-LLM4Climate/docs/multimodal_architecture_diagram.pdf',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("✅ Diagram saved as:")
print("   📊 multimodal_architecture_diagram.png (for presentations)")
print("   📄 multimodal_architecture_diagram.pdf (for formal presentations)")
print("\n🎯 Diagram shows complete multimodal architecture including:")
print("   • Input data specifications and sizes")
print("   • All preprocessing steps (normalization, tokenization)")
print("   • Model components (PrithviWxC encoder, Llama-3 LLM)")
print("   • Location processing and spatial attention")
print("   • Cross-attention fusion mechanism")
print("   • Technical specifications for architecture review")

plt.show()
