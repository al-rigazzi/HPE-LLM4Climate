#!/usr/bin/env python3
# Copyright 2025 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AIFS Multimodal Architecture Diagram Generator (Updated 2025)

This script generates professional architecture diagrams for the AIFS multimodal climate AI system
that combines ECMWF AIFS encoder with Mistral-7B-Instruct-v0.3 language models.

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
- Mistral-7B-Instruct-v0.3 multimodal fusion architecture

Usage:
    python create_aifs_architecture_diagram.py

Output:
    - aifs_multimodal_architecture_diagram.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
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
    "mistral": "#4169E1",  # Royal Blue for Mistral components
    "fusion": "#FF6347",  # Tomato for fusion mechanisms
    "encoder": "#9370DB",  # Medium Purple for encoder
    "data": "#FFD700",  # Gold for data sources
    "output": "#FF69B4",  # Hot Pink for outputs
    "background": "#F8F8FF",  # Ghost White for specs
}


# Helper function to create professional boxes
def create_box(
    axis_handle, xy, width, height, text, color, text_color="white", fontsize=14, fontweight="bold"
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
    axis_handle.add_patch(box)

    # Add text in center of box
    text_x = xy[0] + width / 2
    text_y = xy[1] + height / 2
    axis_handle.text(
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
def create_arrow(axis_handle, start, end, color="black", style="->", linewidth=2):
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
    axis_handle.add_patch(arrow)
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
    "ECMWF AIFS Encoder + Mistral-7B-Instruct-v0.3 Integration",
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
    5,
    1.2,
    "ECMWF Climate Data\n(Zarr/GRIB Format)\n5D Format: [batch, time, ensemble, grid, vars]\n"
    "Example: [1, 2, 1, 542080, 103]\nSurface + 13 Pressure levels",
    colors["data"],
    fontsize=13,
)

# Text Query Input
create_box(
    ax,
    (13.5, 10.5),
    4,
    1.2,
    'Climate Text Queries\n"Analyze temperature patterns"\n"Predict extreme weather"\n'
    "Tokenized: [B, seq_len]\n[2, 512 tokens]",
    colors["data"],
    fontsize=13,
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
    "AIFS Complete Encoder (Pre-trained ECMWF)\nGraphTransformerForwardMapper\n"
    "19.9M parameters (encoder only)\nInput: 103 variables → Raw output: 102 dims\n"
    "Spatial processing: 542,080 grid points",
    colors["aifs"],
    fontsize=13,
)

# =================== MISTRAL PROCESSING LAYER ===================
ax.text(
    13.5,
    9.5,
    "Mistral Processing Pipeline",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# Mistral-7B-Instruct Model
create_box(
    ax,
    (10, 8),
    7.5,
    1.2,
    "Mistral-7B-Instruct-v0.3 Language Model\n7.25B parameters (frozen)\n32 transformer layers\n"
    "4096 hidden dimensions\n32 attention heads",
    colors["mistral"],
    fontsize=13,
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
    "Climate Projection\nAIFS features: 102 → 768/4096\nLinear projection layer\n"
    "Layer normalization\nAdapted for LLM dimension",
    colors["fusion"],
    fontsize=13,
)

# Cross-Modal Fusion
create_box(
    ax,
    (7.5, 5.5),
    4,
    1.3,
    "Fusion Mechanisms\nElement-wise addition\nGated fusion\nCross-attention (optional)\n"
    "Residual connections",
    colors["fusion"],
    fontsize=13,
)

# Location-Aware Processing
create_box(
    ax,
    (12.5, 5.5),
    4.5,
    1.3,
    "Location-Aware Features\nSpatial attention\nGeographic cropping\n"
    "Regional climate analysis\nCoordinate embeddings",
    colors["encoder"],
    fontsize=13,
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
    "Climate Encoder (Configurable)\nAIFS encoder output: 102 dims\n"
    "Projection to LLM space: 102 → 768/4096\nTrainable fusion layers\n"
    "Frozen Mistral-7B-Instruct backbone",
    colors["aifs"],
    fontsize=13,
)

# Integrated Output
create_box(
    ax,
    (7, 3),
    5,
    1.2,
    "Integrated Model Output\nJoint climate-text embeddings\nClimate-aware text generation\n"
    "Multi-task capabilities\nReal-time inference",
    colors["fusion"],
    fontsize=13,
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
    "Climate-Text Analysis Outputs\nClimate-aware text generation\nWeather pattern explanations\n"
    "Scientific insights & predictions\nLocation-specific analysis\nReal-time climate responses",
    colors["output"],
    fontsize=14,
)

# =================== ARROWS ===================
# Data flow arrows from inputs to processing
create_arrow(ax, (3.0, 10.5), (4, 9.2), colors["aifs"])
create_arrow(ax, (15.5, 10.5), (13.2, 9.2), colors["mistral"])

# Processing to fusion
create_arrow(ax, (4, 8), (4.2, 6.8), colors["aifs"])
create_arrow(ax, (13.2, 8), (9.5, 6.8), colors["mistral"])

# Fusion to integration
create_arrow(ax, (4.2, 5.5), (3.5, 4.2), colors["fusion"])
create_arrow(ax, (9.5, 5.5), (9, 4.2), colors["fusion"])
create_arrow(ax, (14.7, 5.5), (10.5, 4.2), colors["encoder"])

# Integration to output
create_arrow(ax, (3.5, 3), (7.5, 1.7), colors["aifs"])
create_arrow(ax, (9.5, 3), (9, 1.7), colors["fusion"])

# =================== TECHNICAL SPECIFICATIONS ===================
# Add technical specs box in bottom right corner
SPECS_TEXT = """AIFS Multimodal Specifications (2025):

AIFS Model: ECMWF AIFS-Single-1.1
Language Model: Mistral-7B-Instruct-v0.3 (7.25B params)
AIFS Encoder: 19.9M parameters (extracted)
Climate Variables: 103 variables
Grid Points: 542,080 spatial points
Framework: PyTorch 2.4+, Python 3.12+
Training: CPU optimized, GPU compatible"""

ax.text(
    0.98,
    0.02,
    SPECS_TEXT,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": colors["background"], "alpha": 0.8},
    color="black",
    fontweight="normal",
)

# =================== LEGEND ===================
legend_elements = [
    patches.Patch(color=colors["aifs"], label="AIFS Components"),
    patches.Patch(color=colors["mistral"], label="Mistral Components"),
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
    xy=(3, 11.1),
    xytext=(5.75, 10.25),
    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.7},
    fontsize=10,
    color="gray",
)

ax.annotate(
    "Direct AIFS Encoding\nNo tokenization step\nDirect feature extraction",
    xy=(4, 8.6),
    xytext=(1, 9.8),
    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.7},
    fontsize=10,
    color="gray",
)

ax.annotate(
    "Climate-Text Fusion\nProjection + Element-wise ops\nFrozen LLM backbone",
    xy=(9, 6.1),
    xytext=(11.5, 7.5),
    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.7},
    fontsize=8,
    color="gray",
)

# Save the diagram (PDF only as requested)
pdf_path = output_dir / "aifs_multimodal_architecture_diagram.pdf"

plt.tight_layout()
plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")

print("AIFS Multimodal Architecture Diagram saved as:")
print(f"   📄 {pdf_path}")
