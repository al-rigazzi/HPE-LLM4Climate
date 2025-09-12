#!/usr/bin/env python3
"""
AIFS Multimodal Architecture Diagram Generator (Updated 2025)

This script generates professional architecture diagrams for the AIFS multimodal climate AI system
that combines ECMWF AIFS encoder with Meta-Llama-3-8B language models.

Key Updates:
- Real AIFS encoder integration (not TimeSeries tokenizer)
- Direct climate data processing pipeline
- Accurate parameter counts and dimensions
- Current fusion mechanisms
- Location-aware processing capabilities

Features:
- Clean, professional styling suitable for presentations
- Technical component specifications and data flow
- Color-coded components by functionality
- Support for PDF output format (PNG removed per request)
- Actual AIFS encoder architecture
- Meta-Llama-3-8B multimodal fusion architecture

Usage:
    python create_aifs_architecture_diagram.py

Output:
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

# Color scheme for AIFS multimodal system (Updated 2025)
colors = {
    "aifs": "#2E8B57",  # Sea Green for AIFS components
    "llama": "#4169E1",  # Royal Blue for Llama components
    "fusion": "#FF6347",  # Tomato for fusion mechanisms
    "encoder": "#9370DB",  # Medium Purple for encoder
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
    "AIFS Multimodal Climate AI Architecture (2025)",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="black",
)
ax.text(
    9,
    12.7,
    "ECMWF AIFS Encoder + Meta-Llama-3-8B Integration",
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
    "ECMWF Climate Data\n(GRIB/Cached Arrays)\n5D Format: [B, T, E, G, V]\nExample: [2, 1, 1, 542080, 103]\nSurface + Pressure levels",
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

# =================== AIFS ENCODER LAYER ===================
ax.text(
    4.5,
    9.5,
    "AIFS Encoder Pipeline",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# AIFS Complete Encoder
create_box(
    ax,
    (0.5, 8),
    7.5,
    1.2,
    "AIFS Complete Encoder (Pre-trained ECMWF)\n• GraphTransformerForwardMapper\n• 19.9M parameters (encoder only)\n• Input: 103 variables → Output: 1024 embeddings\n• Spatial processing: 542,080 grid points",
    colors["aifs"],
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

# Llama 3-8B Model
create_box(
    ax,
    (10, 8),
    7.5,
    1.2,
    "Meta-Llama-3-8B Language Model\n• 8.03B parameters (frozen)\n• 32 transformer layers\n• 4096 hidden dimensions\n• 32 attention heads",
    colors["llama"],
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

# Climate Encoder Projection
create_box(
    ax,
    (2, 5.5),
    4.5,
    1.3,
    "Climate Projection\n• AIFS features: 1024 → 4096\n• Linear projection layer\n• Layer normalization\n• Broadcast to sequence length",
    colors["fusion"],
    fontsize=9,
)

# Cross-Modal Fusion
create_box(
    ax,
    (7.5, 5.5),
    4,
    1.3,
    "Fusion Mechanisms\n• Element-wise addition\n• Gated fusion\n• Cross-attention (optional)\n• Residual connections",
    colors["fusion"],
    fontsize=9,
)

# Location-Aware Processing
create_box(
    ax,
    (12.5, 5.5),
    4.5,
    1.3,
    "Location-Aware Features\n• Spatial attention\n• Geographic cropping\n• Regional climate analysis\n• Coordinate embeddings",
    colors["encoder"],
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

# Climate Encoder Training
create_box(
    ax,
    (1, 3),
    5,
    1.2,
    "Climate Encoder (Trainable)\n• CNN layers + projection\n• 768 → 4096 dimensions\n• Only 2.1M parameters trained\n• Frozen Llama-3-8B backbone",
    colors["aifs"],
    fontsize=9,
)

# Integrated Output
create_box(
    ax,
    (7, 3),
    5,
    1.2,
    "Integrated Model Output\n• Joint climate-text embeddings\n• Climate-aware text generation\n• Multi-task capabilities\n• Real-time inference",
    colors["fusion"],
    fontsize=9,
)

# Performance Metrics
create_box(
    ax,
    (13, 3),
    4,
    1.2,
    "Performance Metrics\n• Memory: 8.5-10.6GB\n• Training: CPU/GPU ready\n• Throughput: 32 samples/s\n• AIFS: 19.9M encoder params",
    "#D3D3D3",  # Light Gray - provides good contrast for black text
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
    "Climate-Text Analysis Outputs\n• Climate-aware text generation\n• Weather pattern explanations\n• Scientific insights & predictions\n• Location-specific analysis\n• Real-time climate responses",
    colors["output"],
    fontsize=10,
)

# =================== ARROWS ===================
# Data flow arrows from inputs to processing
create_arrow(ax, (2.5, 10.5), (4, 9.2), colors["aifs"])
create_arrow(ax, (15.5, 10.5), (13.2, 9.2), colors["llama"])

# Processing to fusion
create_arrow(ax, (4, 8), (4.2, 6.8), colors["aifs"])
create_arrow(ax, (13.2, 8), (9.5, 6.8), colors["llama"])

# Fusion to integration
create_arrow(ax, (4.2, 5.5), (3.5, 4.2), colors["fusion"])
create_arrow(ax, (9.5, 5.5), (9, 4.2), colors["fusion"])
create_arrow(ax, (14.7, 5.5), (14.5, 4.2), colors["encoder"])

# Integration to output
create_arrow(ax, (3.5, 3), (7.5, 1.7), colors["aifs"])
create_arrow(ax, (9.5, 3), (9, 1.7), colors["fusion"])
create_arrow(ax, (15, 3), (10.5, 1.7), colors["background"])

# =================== TECHNICAL SPECIFICATIONS ===================
# Add technical specs box in bottom right corner
specs_text = """AIFS Multimodal Specifications (2025):

• AIFS Model: ECMWF AIFS-Single-1.0
• Language Model: Meta-Llama-3-8B (8.03B params)
• AIFS Encoder: 19.9M parameters (extracted)
• Climate Variables: 103 variables
• Grid Points: 542,080 spatial points
• Memory Usage: 8.5-10.6GB training
• Framework: PyTorch 2.4+, Python 3.12+
• Training: CPU optimized, GPU compatible"""

ax.text(
    0.98,
    0.02,
    specs_text,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["background"], alpha=0.8),
    color="black",
    fontweight="normal",
)

# =================== LEGEND ===================
legend_elements = [
    patches.Patch(color=colors["aifs"], label="AIFS Components"),
    patches.Patch(color=colors["llama"], label="Llama Components"),
    patches.Patch(color=colors["encoder"], label="Encoder/Processing"),
    patches.Patch(color=colors["fusion"], label="Fusion Mechanisms"),
    patches.Patch(color=colors["data"], label="Data Sources"),
    patches.Patch(color=colors["output"], label="Outputs"),
]

ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 0.98), fontsize=10)

# =================== ANNOTATIONS ===================
# Add dimension annotations
ax.annotate(
    "ECMWF Climate Data\n[B, T, E, G, V] format\nCached arrays for fast access",
    xy=(2.5, 11.1),
    xytext=(5, 12),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

ax.annotate(
    "Direct AIFS Encoding\nNo tokenization step\nDirect feature extraction",
    xy=(4, 8.6),
    xytext=(1, 9.8),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

ax.annotate(
    "Climate-Text Fusion\nProjection + Element-wise ops\nFrozen LLM backbone",
    xy=(9, 6.1),
    xytext=(11.5, 7.5),
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    fontsize=8,
    color="gray",
)

# Save the diagram (PDF only as requested)
pdf_path = output_dir / "aifs_multimodal_architecture_diagram.pdf"

plt.tight_layout()
plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")

print("✅ AIFS Multimodal Architecture Diagram saved as:")
print(f"   📄 {pdf_path}")
print("\n🎯 Diagram features (Updated 2025):")
print("   • ECMWF AIFS encoder direct integration")
print("   • Meta-Llama-3-8B language model")
print("   • Real climate data processing pipeline")
print("   • Accurate parameter counts and dimensions")
print("   • Location-aware processing capabilities")
print("   • Professional styling for presentations")

plt.show()
