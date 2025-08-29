#!/usr/bin/env python3
"""
AIFS Multimodal Architecture Diagram Generator

This script generates professional architecture diagrams for the AIFS multimodal climate AI system
that combines ECMWF AIFS TimeSeries tokenization with Llama 3-8B language models.

Features:
- Clean, professional styling suitable for presentations
- Technical component specifications and data flow
- Color-coded components by functionality
- Support for PNG and PDF output formats
- AIFS TimeSeries tokenizer integration
- Llama 3-8B multimodal fusion architecture

Usage:
    python create_aifs_architecture_diagram.py

Output:
    - aifs_multimodal_architecture_diagram.png
    - aifs_multimodal_architecture_diagram.pdf
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
output_dir = script_dir

# Set up the figure with professional styling
plt.style.use("default")
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis("off")

# Color scheme for AIFS multimodal system
colors = {
    "aifs": "#2E8B57",  # Sea Green for AIFS components
    "llama": "#4169E1",  # Royal Blue for Llama components
    "fusion": "#FF6347",  # Tomato for fusion mechanisms
    "tokenizer": "#9370DB",  # Medium Purple for tokenizers
    "data": "#FFD700",  # Gold for data sources
    "output": "#FF69B4",  # Hot Pink for outputs
    "background": "#F8F8FF",  # Ghost White for specs
}


# Helper function to create professional boxes
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


# Helper function to create arrows
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
    9,
    13.2,
    "AIFS Multimodal Climate AI Architecture",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="black",
)
ax.text(
    9,
    12.7,
    "ECMWF AIFS TimeSeries Tokenizer + Llama 3-8B Integration",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="normal",
    color="gray",
)

# =================== INPUT LAYER ===================
ax.text(
    9, 11.8, "Input Layer", ha="center", va="center", fontsize=16, fontweight="bold", color="black"
)

# ECMWF Climate Data Input
create_box(
    ax,
    (0.5, 10.5),
    4,
    1.2,
    "ECMWF Climate Data\n(GRIB/NetCDF)\n5D Tensor: [B, T, V, H, W]\n[2, 4, 73, 16, 16]\nVariables: T, U, V, Q, Z, etc.",
    colors["data"],
    fontsize=9,
)

# Text Query Input
create_box(
    ax,
    (13.5, 10.5),
    4,
    1.2,
    'Climate Text Queries\n"Analyze temperature patterns"\n"Predict extreme weather"\nTokenized: [B, seq_len]\n[2, 512 tokens]',
    colors["data"],
    fontsize=9,
)

# =================== AIFS PROCESSING LAYER ===================
ax.text(
    4.5,
    9.5,
    "AIFS Processing Pipeline",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# AIFS Data Processor
create_box(
    ax,
    (0.5, 8),
    3.5,
    1.2,
    "AIFS Data Processor\nECMWF AIFS-Single-1.0\n• GraphCast architecture\n• 73 variable fields\n• Spatial: 16×16 → 512",
    colors["aifs"],
    fontsize=9,
)

# AIFS TimeSeries Tokenizer
create_box(
    ax,
    (4.5, 8),
    4,
    1.2,
    "AIFS TimeSeries Tokenizer\n• Transformer encoder backbone\n• 5D → Token sequence\n• Hidden dim: 512\n• Temporal modeling: 4 steps",
    colors["tokenizer"],
    fontsize=9,
)

# =================== LLAMA PROCESSING LAYER ===================
ax.text(
    13.5,
    9.5,
    "Llama Processing Pipeline",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# Llama Tokenizer
create_box(
    ax,
    (13.5, 8),
    4,
    1.2,
    "Llama 3-8B Tokenizer\n• BPE encoding\n• 128k vocabulary\n• Special tokens: <|start|>, <|end|>\n• Text → [B, seq_len] tokens",
    colors["tokenizer"],
    fontsize=9,
)

# =================== FUSION LAYER ===================
ax.text(
    9,
    7,
    "Multimodal Fusion Layer",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# Cross-Modal Attention
create_box(
    ax,
    (2, 5.5),
    4.5,
    1.3,
    "Cross-Modal Attention\n• Climate tokens → Llama embeddings\n• Multi-head attention (32 heads)\n• Hidden size: 4096\n• Learnable projection layers",
    colors["fusion"],
    fontsize=9,
)

# Fusion Strategies
create_box(
    ax,
    (7.5, 5.5),
    4,
    1.3,
    "Fusion Strategies\n• Cross-attention\n• Token concatenation\n• Adapter layers\n• Interleaved tokens",
    colors["fusion"],
    fontsize=9,
)

# Advanced Fusion
create_box(
    ax,
    (12.5, 5.5),
    4,
    1.3,
    "Advanced Fusion\n• Temperature-scaled attention\n• Residual connections\n• Layer normalization\n• Dropout regularization",
    colors["fusion"],
    fontsize=9,
)

# =================== MODEL INTEGRATION LAYER ===================
ax.text(
    9,
    4.5,
    "Model Integration Layer",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# Llama 3-8B Model
create_box(
    ax,
    (1, 3),
    5,
    1.2,
    "Llama 3-8B Language Model\n• 32 transformer layers\n• 4096 hidden dimensions\n• 8-bit quantization support\n• Flash attention disabled (compatibility)",
    colors["llama"],
    fontsize=9,
)

# AIFS Encoder
create_box(
    ax,
    (7, 3),
    5,
    1.2,
    "AIFS Encoder Architecture\n• Pre-trained ECMWF weights\n• GraphCast-based design\n• 73-variable processing\n• Autoregressive decoder",
    colors["aifs"],
    fontsize=9,
)

# Integrated Model
create_box(
    ax,
    (13, 3),
    4,
    1.2,
    "Integrated AIFS-Llama\n• Joint embeddings\n• Shared attention\n• End-to-end training\n• Zero-shot capabilities",
    colors["fusion"],
    fontsize=9,
)

# =================== OUTPUT LAYER ===================
ax.text(
    9,
    2,
    "Output Generation Layer",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# Multimodal Outputs - centered
create_box(
    ax,
    (6, 0.5),
    6,
    1.2,
    "Multimodal Climate Analysis\n• Scientific explanations\n• Weather predictions\n• Trend analysis\n• Uncertainty quantification\n• Interactive responses",
    colors["output"],
    fontsize=10,
)

# =================== ARROWS ===================
# Data flow arrows
create_arrow(ax, (2.5, 10.5), (2.2, 9.2), colors["aifs"])
create_arrow(ax, (15.5, 10.5), (15.5, 9.2), colors["llama"])

# Processing flow
create_arrow(ax, (4, 8), (6.5, 8.6), colors["aifs"])
create_arrow(ax, (6.5, 8), (4.2, 6.8), colors["tokenizer"])
create_arrow(ax, (15.5, 8), (14, 6.8), colors["tokenizer"])

# Fusion connections
create_arrow(ax, (4.2, 5.5), (3.5, 4.2), colors["fusion"])
create_arrow(ax, (9.5, 5.5), (9.2, 4.2), colors["fusion"])
create_arrow(ax, (14.2, 5.5), (14.5, 4.2), colors["fusion"])

# Output generation
create_arrow(ax, (3.5, 3), (7.5, 1.7), colors["llama"])
create_arrow(ax, (9.5, 3), (9, 1.7), colors["aifs"])
create_arrow(ax, (15, 3), (10.5, 1.7), colors["fusion"])

# =================== TECHNICAL SPECIFICATIONS ===================
# Add technical specs box in bottom left corner
specs_text = """AIFS Multimodal Specifications:

• AIFS Model: ECMWF AIFS-Single-1.0
• Language Model: Meta-Llama-3-8B
• Climate Variables: 73 fields
• Spatial Resolution: 0.25° × 0.25°
• Temporal Resolution: 3-hourly
• Context Length: 8192 tokens
• Fusion Method: Cross-attention"""

create_box(
    ax,
    (0.5, 0.5),
    4.5,
    2.2,
    specs_text,
    colors["background"],
    "black",
    fontsize=8,
    fontweight="normal",
)

# =================== LEGEND ===================
legend_elements = [
    patches.Patch(color=colors["aifs"], label="AIFS Components"),
    patches.Patch(color=colors["llama"], label="Llama Components"),
    patches.Patch(color=colors["tokenizer"], label="Tokenizers"),
    patches.Patch(color=colors["fusion"], label="Fusion Mechanisms"),
    patches.Patch(color=colors["data"], label="Data Sources"),
    patches.Patch(color=colors["output"], label="Outputs"),
]

ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 0.98), fontsize=10)

# =================== ANNOTATIONS ===================
# Add dimension annotations
ax.annotate(
    "5D Climate Tensor\nB=Batch, T=Time, V=Variables\nH=Height, W=Width",
    xy=(2.5, 11.1),
    xytext=(5, 12),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

ax.annotate(
    "TimeSeries Tokenization\nSequential → Embeddings",
    xy=(6.5, 8.6),
    xytext=(9, 9.8),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

ax.annotate(
    "Cross-Modal Fusion\nClimate ↔ Language",
    xy=(9, 6.1),
    xytext=(11.5, 7.5),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

# Save the diagram
png_path = output_dir / "aifs_multimodal_architecture_diagram.png"
pdf_path = output_dir / "aifs_multimodal_architecture_diagram.pdf"

plt.tight_layout()
plt.savefig(str(png_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")

print("✅ AIFS Multimodal Architecture Diagram saved as:")
print(f"   📊 {png_path}")
print(f"   📄 {pdf_path}")
print("\n🎯 Diagram features:")
print("   • ECMWF AIFS TimeSeries tokenizer integration")
print("   • Llama 3-8B language model components")
print("   • Cross-modal attention fusion mechanisms")
print("   • 5D climate data processing pipeline")
print("   • Technical specifications for AIFS system")
print("   • Professional styling for presentations")

plt.show()
