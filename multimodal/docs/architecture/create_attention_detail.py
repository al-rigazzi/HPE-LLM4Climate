#!/usr/bin/env python3
"""
Detailed Cross-Attention Mechanism Diagram

This script creates a focused diagram showing the internal workings
of the cross-attention fusion mechanism for executive technical review.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Rectangle

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
output_dir = script_dir

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Color scheme
colors = {
    "query": "#4169E1",  # Royal Blue (Text)
    "key": "#2E8B57",  # Sea Green (Climate)
    "value": "#2E8B57",  # Sea Green (Climate)
    "attention": "#FF6347",  # Tomato (Attention)
    "output": "#9370DB",  # Medium Purple (Output)
    "math": "#FFD700",  # Gold (Math operations)
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
        linewidth=2,
        alpha=0.9,
    )
    ax.add_patch(box)

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


def create_arrow(ax, start, end, color="black", style="->", linewidth=2):
    """Create professional arrows between components"""
    arrow = ConnectionPatch(
        start,
        end,
        "data",
        "data",
        arrowstyle=style,
        shrinkA=5,
        shrinkB=5,
        mutation_scale=20,
        fc=color,
        ec=color,
        linewidth=linewidth,
    )
    ax.add_patch(arrow)
    return arrow


# Title
ax.text(
    7,
    9.5,
    "Cross-Attention Fusion Mechanism Detail",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# Input representations
ax.text(7, 8.7, "Input Representations", ha="center", va="center", fontsize=12, fontweight="bold")

# Text Query (Q)
create_box(
    ax,
    (1, 7.8),
    2.5,
    0.6,
    "Text Embeddings (Q)\n[B, 512, 4096]\nLlama-3 output",
    colors["query"],
    fontsize=9,
)

# Climate Key (K)
create_box(
    ax,
    (5.75, 7.8),
    2.5,
    0.6,
    "Climate Features (K)\n[B, 1024, 4096]\nProjected PrithviWxC",
    colors["key"],
    fontsize=9,
)

# Climate Value (V)
create_box(
    ax,
    (10.5, 7.8),
    2.5,
    0.6,
    "Climate Features (V)\n[B, 1024, 4096]\nProjected PrithviWxC",
    colors["value"],
    fontsize=9,
)

# Multi-Head Attention Computation
ax.text(
    7,
    6.8,
    "Multi-Head Attention Computation",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
)

# Linear projections
create_box(
    ax,
    (1, 5.8),
    2.5,
    0.5,
    "Linear Projection\nWq · Q\n[4096 → 4096]",
    colors["math"],
    "black",
    fontsize=8,
)

create_box(
    ax,
    (5.75, 5.8),
    2.5,
    0.5,
    "Linear Projection\nWk · K\n[4096 → 4096]",
    colors["math"],
    "black",
    fontsize=8,
)

create_box(
    ax,
    (10.5, 5.8),
    2.5,
    0.5,
    "Linear Projection\nWv · V\n[4096 → 4096]",
    colors["math"],
    "black",
    fontsize=8,
)

# Head splitting
create_box(
    ax,
    (4, 4.8),
    6,
    0.5,
    "Split into 32 Attention Heads\nEach head: [B, seq_len, 128] (4096/32 = 128)",
    colors["attention"],
    fontsize=9,
)

# Attention computation
create_box(
    ax,
    (4, 3.6),
    6,
    0.8,
    "Scaled Dot-Product Attention\nAttention(Q,K,V) = softmax(QK^T/√d_k)V\nd_k = 128 (per head)",
    colors["attention"],
    fontsize=9,
)

# Mathematical formula
ax.text(
    7, 2.8, "Attention Score Computation:", ha="center", va="center", fontsize=11, fontweight="bold"
)
ax.text(
    7,
    2.4,
    r"$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$",
    ha="center",
    va="center",
    fontsize=12,
    style="italic",
)
ax.text(
    7,
    2.0,
    "Where Q=[B,512,128], K=[B,1024,128], V=[B,1024,128] per head",
    ha="center",
    va="center",
    fontsize=9,
    color="gray",
)

# Output processing
ax.text(7, 1.4, "Output Processing", ha="center", va="center", fontsize=12, fontweight="bold")

create_box(ax, (3, 0.6), 3, 0.5, "Concatenate Heads\n[B, 512, 4096]", colors["output"], fontsize=9)

create_box(
    ax,
    (8, 0.6),
    3,
    0.5,
    "Output Projection\nWo · concat\n[4096 → 4096]",
    colors["output"],
    fontsize=9,
)

# Add arrows showing data flow
create_arrow(ax, (2.25, 7.8), (2.25, 6.3), colors["query"])
create_arrow(ax, (7, 7.8), (7, 6.3), colors["key"])
create_arrow(ax, (11.75, 7.8), (11.75, 6.3), colors["value"])

create_arrow(ax, (2.25, 5.8), (4, 5.05), colors["query"])
create_arrow(ax, (7, 5.8), (7, 5.3), colors["key"])
create_arrow(ax, (11.75, 5.8), (10, 5.05), colors["value"])

create_arrow(ax, (7, 4.8), (7, 4.4), colors["attention"])
create_arrow(ax, (7, 3.6), (7, 1.9), colors["attention"])

create_arrow(ax, (6, 1.6), (4.5, 1.1), colors["output"])
create_arrow(ax, (8, 1.6), (9.5, 1.1), colors["output"])

# Technical specifications box
specs_text = """Technical Implementation Details:
• Attention Mechanism: Multi-Head Cross-Attention
• Number of Heads: 32 (h=32)
• Model Dimension: 4096 (d_model=4096)
• Head Dimension: 128 (d_k = d_model/h = 128)
• Query Source: Text embeddings from Llama-3
• Key/Value Source: Climate features from PrithviWxC
• Attention Matrix Size: [512, 1024] per sample
• Output: Climate-conditioned text representations
• Residual Connections: Applied after attention
• Layer Normalization: Applied before and after"""

create_box(ax, (0.5, 0.2), 5, 3.2, specs_text, "#F8F8FF", "black", fontsize=8, fontweight="normal")

# Attention pattern visualization
ax.text(
    11.5,
    4,
    "Attention Pattern\nVisualization",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

# Create a small heatmap showing attention pattern
attention_data = np.random.rand(8, 12) * 0.8 + 0.1  # Simulated attention weights
im = ax.imshow(attention_data, extent=[9.5, 13, 2.5, 4], cmap="Reds", alpha=0.8, aspect="auto")

# Add labels for the heatmap
ax.text(11.25, 4.2, "Text Tokens", ha="center", va="center", fontsize=8, rotation=90)
ax.text(11.25, 2.3, "Climate Patches", ha="center", va="center", fontsize=8)

# Save the detailed diagram in the current directory
png_path = output_dir / "cross_attention_detail.png"
pdf_path = output_dir / "cross_attention_detail.pdf"

plt.tight_layout()
plt.savefig(str(png_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")

print("✅ Detailed cross-attention diagram saved as:")
print(f"   📊 {png_path}")
print(f"   📄 {pdf_path}")
print("\n🔍 Detailed diagram shows:")
print("   • Mathematical formulation of attention mechanism")
print("   • Multi-head attention computation steps")
print("   • Tensor dimensions at each step")
print("   • Technical implementation parameters")
print("   • Attention pattern visualization")

plt.show()
