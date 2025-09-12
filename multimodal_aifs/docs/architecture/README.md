# AIFS Multimodal Architecture Diagrams (Updated 2025)

This directory contains tools for generating professional architecture diagrams for the AIFS multimodal climate AI system that combines ECMWF AIFS encoder with Meta-Llama-3-8B language models.

## Available Diagrams

### 1. Main Architecture Diagram
**File**: `create_aifs_architecture_diagram.py`

Generates comprehensive system architecture showing:
- ECMWF climate data ingestion (5D tensors)
- AIFS encoder direct integration (19.9M parameters)
- Meta-Llama-3-8B integration (8.03B parameters frozen)
- Climate-text fusion mechanisms
- Location-aware processing capabilities
- Accurate technical specifications

**Output**: `aifs_multimodal_architecture_diagram.pdf`

### 2. Cross-Attention Detail Diagram
**File**: `create_aifs_attention_detail.py`

Generates detailed technical diagram of the attention mechanism:
- Mathematical formulation of cross-attention
- Tensor dimension tracking (1024 → 4096 projection)
- Multi-head attention computation (32 heads)
- AIFS-Llama fusion details
- Performance specifications

**Output**: `aifs_cross_attention_detail.pdf`

## Usage

### Generate All Diagrams
From the project root:
```bash
cd multimodal_aifs/docs/architecture
python create_aifs_architecture_diagram.py
python create_aifs_attention_detail.py
```

### Individual Generation
```bash
# Main architecture
python create_aifs_architecture_diagram.py

# Attention mechanism detail
python create_aifs_attention_detail.py
```

## Output Files

All diagrams are generated in PDF format for professional presentations and documentation:

- **PDF files**: Vector format for printing and executive documents, high-quality scaling

### File Naming Convention
- `aifs_multimodal_architecture_diagram.pdf` - Complete system overview
- `aifs_cross_attention_detail.pdf` - Detailed attention mechanism

## Technical Details (Updated 2025)

### AIFS Architecture Features
- **Data Source**: ECMWF climate data (GRIB/Cached Arrays)
- **Model**: AIFS-Single-1.0 (Encoder extracted, 19.9M parameters)
- **Processing**: Direct encoder integration (no tokenization)
- **Integration**: Meta-Llama-3-8B language model (8.03B parameters)
- **Fusion**: Element-wise addition, gated fusion, cross-attention
- **Memory**: 8.5-10.6GB training, CPU optimized

### Diagram Features
- Professional styling suitable for technical presentations
- Color-coded components by functionality
- Mathematical formulations and tensor dimensions
- Performance and memory specifications
- Technical annotations and flow indicators

### Dependencies
- matplotlib
- numpy
- pathlib (standard library)

The diagrams are designed for:
- Technical team presentations
- Executive reviews
- Research paper illustrations
- Documentation and training materials
