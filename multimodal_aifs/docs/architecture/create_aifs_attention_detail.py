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
AIFS Cross-Attention Detail Diagram Generator (Updated 2025)

This script generates detailed technical diagrams of the attention mechanism
used in the AIFS multimodal climate AI system for climate-text fusion.

Key Updates:
- Real AIFS encoder dimensions (102 raw → 218 projected embeddings)
- Accurate Mistral-7B-Instruct-v0.3 specifications
- Current fusion mechanisms and projection layers
- Actual parameter counts

Features:
- Mathematical formulation of cross-attention
- Tensor dimension tracking through pipeline
- Professional styling for technical presentations

Usage:
    python create_aifs_attention_detail.py

Output:
    - aifs_cross_attention_detail.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

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
    "mistral_tokens": "#4169E1",  # Royal Blue for Mistral tokens
    "attention": "#FF6347",  # Tomato for attention computation
    "projection": "#9370DB",  # Medium Purple for projections
    "math": "#FFD700",  # Gold for mathematical operations
    "output": "#FF69B4",  # Hot Pink for outputs
    "background": "#F8F8FF",  # Ghost White for specs
}


def create_box(
    axis_handle, xy, width, height, text, color, text_color="white", fontsize=14, fontweight="bold"
):
    """Create a professional rounded rectangle box with text"""
    text_box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
    )
    axis_handle.add_patch(text_box)

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

    return text_box


def create_arrow(axis_handle, start, end, color="black", style="->", linewidth=2, alpha=1.0):
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
    axis_handle.add_patch(arrow)
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
    11.2,
    "TimeSeries Tokens ↔ Mistral-7B-Instruct Embeddings Fusion",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="normal",
    color="gray",
)

# =================== MATHEMATICAL FORMULATION ===================
ax.text(
    8,
    10.8,
    "Mathematical Formulation",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# Main attention formula
MATH_TEXT = """Multi-Head Cross-Attention:
1. Projection: Q = X_text W_Q,  K = V = X'_climate W_K,V
2. Multi-Head: Q_h = Q W_h^Q,  K_h = K W_h^K,  V_h = V W_h^V
3. Attention: A_h = softmax(Q_h K_h^T / √(d_k))
4. Output: O_h = A_h V_h
5. Concatenate: O = Concat(O_1, ..., O_32) W_O
6. Residual: Y = LayerNorm(X_text + O)"""

create_box(
    ax, (0.5, 9.0), 7, 1.5, MATH_TEXT, colors["math"], "black", fontsize=13, fontweight="normal"
)

# =================== ATTENTION COMPUTATION DETAIL ===================
# Attention matrix visualization
create_box(
    ax,
    (8.0, 9.0),
    3.5,
    1.5,
    "Attention Matrix\nA: [B, 32, 128, 64]\nText "
    "pos × Climate pos\nTemperature-scaled\nτ = 0.1 (learnable)",
    colors["attention"],
    fontsize=13,
)

# =================== INPUT TOKEN REPRESENTATIONS ===================
ax.text(
    8,
    8.5,
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
    (0.5, 7),
    3.5,
    1,
    "AIFS Climate Embeddings\nX_climate = [B, T, 218]\nB=1, "
    "time=2, d_model=218\nFrom AIFS encoder output",
    colors["aifs_tokens"],
    fontsize=13,
)

# Mistral Tokens
create_box(
    ax,
    (11.5, 7),
    4,
    1,
    "LLM Text Tokens\nX_text = [B, seq_len, d_llm]\nB=1, "
    "seq_len variable, d_llm=4096\nFrom climate queries",
    colors["mistral_tokens"],
    fontsize=13,
)

# =================== PROJECTION LAYER ===================
ax.text(
    8,
    6.3,
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
    (0.5, 5),
    3.5,
    0.8,
    "Climate Projector\nW_c: 218 → d_llm\nLinear(218, 4096) + LayerNorm",
    colors["projection"],
    fontsize=13,
)

# Projected Climate Tokens
create_box(
    ax,
    (5, 5),
    3,
    0.8,
    "Projected Climate\nX'_climate = [B, T, d_llm]\nAligned with LLM dim",
    colors["projection"],
    fontsize=13,
)

# =================== MULTI-HEAD ATTENTION ===================
ax.text(
    8,
    4,
    "Multi-Head Cross-Attention Computation",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="black",
)

# Query, Key, Value projections
# Query, Key, Value projections
create_box(
    ax,
    (0.5, 2.8),
    2.3,
    0.8,
    "Query Projection\nQ = X_text · W_Q\nQ: [B, 128, 4096]",
    colors["attention"],
    fontsize=11,
)

create_box(
    ax,
    (3, 2.8),
    2.3,
    0.8,
    "Key Projection\nK = X'_climate · W_K\nK: [B, 64, 4096]",
    colors["attention"],
    fontsize=11,
)

create_box(
    ax,
    (5.5, 2.8),
    2.3,
    0.8,
    "Value Projection\nV = X'_climate · W_V\nV: [B, 64, 4096]",
    colors["attention"],
    fontsize=11,
)

create_box(
    ax,
    (8.2, 2.8),
    3,
    0.8,
    "Multi-Head Split\n32 heads of d_k=128\nParallel computation",
    colors["attention"],
    fontsize=11,
)

create_box(
    ax,
    (12, 2.8),
    3.5,
    0.8,
    "Attention Computation\nA_h = softmax(Q_h K_h^T / √d_k)\nPer-head attention weights",
    colors["attention"],
    fontsize=11,
)


# =================== OUTPUT FUSION ===================
ax.text(
    8,
    1.9,
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
    (0.5, 1.1),
    3,
    0.6,
    "Head Concatenation\nConcat(O_1,...,O_32)\n[B, 128, 4096]",
    colors["output"],
    fontsize=13,
)

# Output projection
create_box(
    ax,
    (4, 1.1),
    3,
    0.6,
    "Output Projection\nW_O: 4096 → 4096\nLinear + Dropout",
    colors["output"],
    fontsize=13,
)

# Final output
create_box(
    ax,
    (7.5, 1.1),
    4,
    0.6,
    "Fused Embeddings\nY = [B, 128, 4096]\nText enhanced with climate context"
    "\nReady for Mistral decoder",
    colors["output"],
    fontsize=13,
)

# =================== ARROWS ===================
# Input flow
create_arrow(ax, (2.25, 7), (2.25, 5.8), colors["aifs_tokens"])
create_arrow(ax, (3.25, 5.4), (5, 5.4), colors["projection"])

# Projection flow
create_arrow(ax, (6.6, 4.95), (1.6, 3.6), colors["projection"])  # To Q
create_arrow(ax, (6.6, 4.95), (4.1, 3.6), colors["projection"])  # To K
create_arrow(ax, (6.6, 4.95), (6.6, 3.6), colors["projection"])  # To V

create_arrow(ax, (13.5, 7), (1.6, 3.6), colors["mistral_tokens"])  # Text to Q

# Attention computation flow
create_arrow(ax, (7.8, 2.8), (8.5, 2.8), colors["attention"])  # To Multi-head split
create_arrow(ax, (11.2, 2.8), (12, 2.8), colors["attention"])  # To Attention computation

# Output flow
create_arrow(ax, (13.75, 2.8), (2, 1.7), colors["attention"])  # From attention to concatenation
create_arrow(ax, (3.5, 1.4), (4, 1.4), colors["output"])  # Concat to projection
create_arrow(ax, (7, 1.4), (7.5, 1.4), colors["output"])  # Projection to fused

# =================== TECHNICAL ANNOTATIONS ===================
# Dimension annotations
ax.annotate(
    "Climate tokens\nprojected to\nLLM dimension",
    xy=(6.5, 5.65),
    xytext=(9, 6.5),
    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.7},
    fontsize=11,
    color="gray",
)

ax.annotate(
    "32 attention heads\nparallel computation",
    xy=(9.7, 3.6),
    xytext=(10.5, 4.5),
    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.7},
    fontsize=11,
    color="gray",
)

ax.annotate(
    "Cross-modal\nattention matrix",
    xy=(9.75, 9.25),
    xytext=(12.5, 8.5),
    arrowprops={"arrowstyle": "->", "color": "gray", "alpha": 0.7},
    fontsize=11,
    color="gray",
)

# =================== SPECIFICATIONS PANEL ===================
SPECS_TEXT = """AIFS Cross-Attention Specifications:

Input Dimensions:
  - Climate: [B, T, 218] → projected to [B, T, 4096]
  - Text: [B, seq_len, 4096] (Mistral-7B dimension)

Multi-Head Configuration:
  - Heads: 32 (Mistral-7B-Instruct)
  - Per-head dimension: 128 (4096 ÷ 32)
  - Total parameters: ~67M for attention layers

Attention Mechanism:
  - Query: from text embeddings
  - Key/Value: from projected climate embeddings
  - Temperature scaling: learnable τ ∈ [0.01, 1.0]
  - Dropout: 0.1 during training"""

# Create the box
box = FancyBboxPatch(
    (12.2, 9.0),
    3.6,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor=colors["background"],
    edgecolor="black",
    linewidth=1.5,
    alpha=0.9,
)
ax.add_patch(box)

# Add left-aligned text
ax.text(
    12.2,  # Left margin
    11.5,  # Near top of box (box top is at 9.0 + 2.5 = 11.5)
    SPECS_TEXT,
    ha="left",
    va="top",
    fontsize=10,
    fontweight="normal",
    color="black",
)

# Save the diagram (PDF only as requested)
pdf_path = output_dir / "aifs_cross_attention_detail.pdf"

plt.tight_layout()
plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")

print(f"AIFS Cross-Attention Detail Diagram saved as: {pdf_path}")
