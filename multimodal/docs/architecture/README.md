# Architecture Diagram Tools

This directory contains tools for generating architecture diagrams for the HPE LLM4Climate multimodal system.

## Contents

- `create_architecture_diagram.py` - Generates comprehensive multimodal architecture diagrams
- `create_attention_detail.py` - Creates detailed cross-attention mechanism diagrams

## Usage

### Generate Architecture Diagram

```bash
cd multimodal/docs/architecture
python create_architecture_diagram.py
```

This will generate:
- `multimodal_architecture_diagram.png` - High-resolution PNG for presentations
- `multimodal_architecture_diagram.pdf` - PDF format for executive documents

### Generate Cross-Attention Detail

```bash
cd multimodal/docs/architecture
python create_attention_detail.py
```

This will generate:
- `cross_attention_detail.png` - Detailed attention mechanism PNG
- `cross_attention_detail.pdf` - PDF format for technical review

## Features

### Architecture Diagram
- Complete multimodal system overview
- Input/output specifications with tensor dimensions
- Processing pipeline visualization
- Technical specifications box
- Professional color coding by component type

### Attention Detail Diagram
- Mathematical formulation of cross-attention
- Multi-head attention computation steps
- Tensor dimension tracking
- Technical implementation details
- Attention pattern visualization

## Requirements

```bash
pip install matplotlib numpy
```

## Output Location

All generated diagrams are saved in the same directory as the scripts (`multimodal/docs/architecture/`).

## Customization

Both scripts use configurable:
- Color schemes for different component types
- Figure sizes and DPI settings
- Technical specifications and parameters
- Mathematical notation and formulas

## Integration

These tools are designed to work from within the multimodal framework and automatically handle path resolution for output files.
