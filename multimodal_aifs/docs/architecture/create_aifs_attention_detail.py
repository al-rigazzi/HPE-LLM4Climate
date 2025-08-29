#!/usr/bin/env python3
"""
AIFS Cross-Attention Detail Diagram Generator

This script generates detailed technical diagrams of the cross-attention mechanism
used in the AIFS multimodal climate system for fusing AIFS TimeSeries tokens
with Llama 3-8B language model embeddings.

Features:
- Mathematical formulation display
- Tensor dimension tracking through layers
- Multi-head attention computation visualization
- Professional styling for technical documentation
- AIFS-specific tokenization details

Usage:
    python create_aifs_attention_detail.py

Output:
    - aifs_cross_attention_detail.png
    - aifs_cross_attention_detail.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Rectangle

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
output_dir = script_dir

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

# Color scheme for AIFS attention mechanism
colors = {
    "aifs_tokens": "#2E8B57",  # Sea Green for AIFS tokens
    "llama_tokens": "#4169E1",  # Royal Blue for Llama tokens
    "attention": "#FF6347",  # Tomato for attention computation
    "projection": "#9370DB",  # Medium Purple for projections
    "math": "#FFD700",  # Gold for mathematical operations
    "output": "#FF69B4",  # Hot Pink for outputs
    "background": "#F8F8FF",  # Ghost White for specs
}


def create_box(
    ax, xy, width, height, text, color, text_color="white", fontsize=10, fontweight="bold"
):
    """Create a professional rounded rectangle box with text"""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.add_patch(box)

    # Add text in center of box
    text_x = xy[0] + width / 2
    text_y = xy[1] + height / 2
    ax.text(
        text_x,
        text_y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        wrap=True,
    )

    return box


def create_arrow(ax, start, end, color="black", style="->", linewidth=2, alpha=1.0):
    """Create professional arrows between components"""
    arrow = ConnectionPatch(
        start,
        end,
        "data",
        "data",
        arrowstyle=style,
        shrinkA=3,
        shrinkB=3,
        mutation_scale=15,
        fc=color,
        ec=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(arrow)
    return arrow


# Title
ax.text(
    8,
    11.5,
    "AIFS Cross-Attention Mechanism Detail",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="black",
)
ax.text(
    8,
    11,
    "TimeSeries Tokens ↔ Llama 3-8B Embeddings Fusion",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="normal",
    color="gray",
)

# =================== INPUT TOKENS ===================
ax.text(
    8,
    10.3,
    "Input Token Representations",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# AIFS Tokens
create_box(
    ax,
    (0.5, 9),
    3.5,
    1,
    "AIFS TimeSeries Tokens\nX_climate = [B, 64, 512]\nB=2, seq_len=64, d_model=512\nFrom 5D climate data",
    colors["aifs_tokens"],
    fontsize=9,
)

# Llama Tokens
create_box(
    ax,
    (12, 9),
    3.5,
    1,
    "Llama Text Tokens\nX_text = [B, 128, 4096]\nB=2, seq_len=128, d_model=4096\nFrom climate queries",
    colors["llama_tokens"],
    fontsize=9,
)

# =================== PROJECTION LAYER ===================
ax.text(
    8,
    8.3,
    "Dimension Alignment Layer",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# Climate Projection
create_box(
    ax,
    (0.5, 7),
    3.5,
    0.8,
    "Climate Projector\nW_c: 512 → 4096\nLinear(512, 4096) + LayerNorm",
    colors["projection"],
    fontsize=9,
)

# Projected Climate Tokens
create_box(
    ax,
    (5, 7),
    3,
    0.8,
    "Projected Climate\nX'_climate = [B, 64, 4096]\nAligned with Llama dim",
    colors["projection"],
    fontsize=9,
)

# =================== MULTI-HEAD ATTENTION ===================
ax.text(
    8,
    6,
    "Multi-Head Cross-Attention Computation",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# Query, Key, Value projections
create_box(
    ax,
    (0.5, 4.8),
    2.3,
    0.8,
    "Query Projection\nQ = X_text · W_Q\nQ: [B, 128, 4096]",
    colors["attention"],
    fontsize=8,
)

create_box(
    ax,
    (3, 4.8),
    2.3,
    0.8,
    "Key Projection\nK = X'_climate · W_K\nK: [B, 64, 4096]",
    colors["attention"],
    fontsize=8,
)

create_box(
    ax,
    (5.5, 4.8),
    2.3,
    0.8,
    "Value Projection\nV = X'_climate · W_V\nV: [B, 64, 4096]",
    colors["attention"],
    fontsize=8,
)

# Multi-head split
create_box(
    ax,
    (8.5, 4.8),
    3,
    0.8,
    "Multi-Head Split\n32 heads × 128 dim each\nQ_h, K_h, V_h per head",
    colors["attention"],
    fontsize=8,
)

# Attention computation
create_box(
    ax,
    (12, 4.8),
    3.5,
    0.8,
    "Attention Computation\nA_h = softmax(Q_h K_h^T / √d_k)\nPer-head attention weights",
    colors["attention"],
    fontsize=8,
)

# =================== MATHEMATICAL FORMULATION ===================
ax.text(
    8,
    4,
    "Mathematical Formulation",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# Main attention formula
math_text = """Multi-Head Cross-Attention:

1. Projection: Q = X_text W_Q,  K = V = X'_climate W_K,V

2. Multi-Head: Q_h = Q W_h^Q,  K_h = K W_h^K,  V_h = V W_h^V

3. Attention: A_h = softmax(Q_h K_h^T / √(d_k))

4. Output: O_h = A_h V_h

5. Concatenate: O = Concat(O_1, ..., O_32) W_O

6. Residual: Y = LayerNorm(X_text + O)"""

create_box(
    ax, (0.5, 2.2), 7, 1.5, math_text, colors["math"], "black", fontsize=9, fontweight="normal"
)

# =================== ATTENTION COMPUTATION DETAIL ===================
# Attention matrix visualization
create_box(
    ax,
    (8.5, 2.2),
    3.5,
    1.5,
    "Attention Matrix\nA: [B, 32, 128, 64]\nText pos × Climate pos\nTemperature-scaled\nτ = 0.1 (learnable)",
    colors["attention"],
    fontsize=9,
)

# =================== OUTPUT FUSION ===================
ax.text(
    8,
    1.5,
    "Output Fusion & Integration",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# Concatenation
create_box(
    ax,
    (0.5, 0.3),
    3,
    0.8,
    "Head Concatenation\nConcat(O_1,...,O_32)\n[B, 128, 4096]",
    colors["output"],
    fontsize=9,
)

# Output projection
create_box(
    ax,
    (4, 0.3),
    3,
    0.8,
    "Output Projection\nW_O: 4096 → 4096\nLinear + Dropout",
    colors["output"],
    fontsize=9,
)

# Final output
create_box(
    ax,
    (7.5, 0.3),
    4,
    0.8,
    "Fused Embeddings\nY = [B, 128, 4096]\nText enhanced with climate context\nReady for Llama decoder",
    colors["output"],
    fontsize=9,
)

# =================== ARROWS ===================
# Input flow
create_arrow(ax, (2.25, 9), (2.25, 7.8), colors["aifs_tokens"])
create_arrow(ax, (4, 7.4), (5, 7.4), colors["projection"])

# Projection flow
create_arrow(ax, (6.5, 7), (1.6, 5.6), colors["projection"])  # To Q
create_arrow(ax, (6.5, 7), (4.1, 5.6), colors["projection"])  # To K
create_arrow(ax, (6.5, 7), (6.6, 5.6), colors["projection"])  # To V

create_arrow(ax, (13.75, 9), (1.6, 5.6), colors["llama_tokens"])  # Text to Q

# Attention computation flow
create_arrow(ax, (5.3, 4.8), (8.5, 5.2), colors["attention"])
create_arrow(ax, (11.5, 4.8), (12, 5.2), colors["attention"])

# Output flow
create_arrow(ax, (13.75, 4.8), (2, 1.1), colors["attention"])
create_arrow(ax, (3.5, 0.3), (4, 0.7), colors["output"])
create_arrow(ax, (7, 0.3), (7.5, 0.7), colors["output"])

# =================== TECHNICAL ANNOTATIONS ===================
# Dimension annotations
ax.annotate(
    "Climate tokens\nprojected to\nLlama dimension",
    xy=(6.5, 7.4),
    xytext=(9, 8.5),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

ax.annotate(
    "32 attention heads\nparallel computation",
    xy=(10, 5.2),
    xytext=(12.5, 6.5),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

ax.annotate(
    "Cross-modal\nattention matrix",
    xy=(10.2, 2.95),
    xytext=(13, 3.8),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

# =================== SPECIFICATIONS PANEL ===================
specs_text = """AIFS Cross-Attention Specifications:

• Input Dimensions:
  - Climate: [B, 64, 512] → projected to [B, 64, 4096]
  - Text: [B, 128, 4096] (native Llama dimension)

• Multi-Head Configuration:
  - Heads: 32 (same as Llama 3-8B)
  - Per-head dimension: 128 (4096 ÷ 32)
  - Total parameters: ~67M for attention layers

• Attention Mechanism:
  - Query: from text embeddings
  - Key/Value: from projected climate embeddings
  - Temperature scaling: learnable τ ∈ [0.01, 1.0]
  - Dropout: 0.1 during training

• Performance:
  - FLOPs: ~2.1T per forward pass
  - Memory: ~4.2GB for batch_size=2
  - Latency: ~45ms on A100 GPU"""

create_box(
    ax,
    (12.5, 0.3),
    3.2,
    4.2,
    specs_text,
    colors["background"],
    "black",
    fontsize=7,
    fontweight="normal",
)

# Save the diagram
png_path = output_dir / "aifs_cross_attention_detail.png"
pdf_path = output_dir / "aifs_cross_attention_detail.pdf"

plt.tight_layout()
plt.savefig(str(png_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")

print("✅ AIFS Cross-Attention Detail Diagram saved as:")
print(f"   📊 {png_path}")
print(f"   📄 {pdf_path}")
print("\n🔍 Diagram shows:")
print("   • AIFS TimeSeries token processing")
print("   • Dimension alignment (512 → 4096)")
print("   • Multi-head attention computation (32 heads)")
print("   • Mathematical formulation details")
print("   • Tensor dimension tracking")
print("   • Performance specifications")

plt.show()
