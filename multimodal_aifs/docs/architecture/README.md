# AIFS Multimodal Architecture Diagrams

This directory contains tools for generating professional architecture diagrams for the AIFS multimodal climate AI system that combines ECMWF AIFS TimeSeries tokenization with Llama 3-8B language models.

## Available Diagrams

### 1. Main Architecture Diagram
**File**: `create_aifs_architecture_diagram.py`

Generates comprehensive system architecture showing:
- ECMWF climate data ingestion (5D tensors)
- AIFS TimeSeries tokenizer processing
- Llama 3-8B integration
- Cross-modal fusion mechanisms
- Technical specifications

**Output**: `aifs_multimodal_architecture_diagram.{png,pdf}`

### 2. Cross-Attention Detail Diagram
**File**: `create_aifs_attention_detail.py`

Generates detailed technical diagram of the attention mechanism:
- Mathematical formulation of cross-attention
- Tensor dimension tracking (512 → 4096 projection)
- Multi-head attention computation (32 heads)
- AIFS-Llama token fusion details
- Performance specifications

**Output**: `aifs_cross_attention_detail.{png,pdf}`

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

All diagrams are generated in both PNG and PDF formats:

- **PNG files**: High-resolution (300 DPI) for presentations and web use
- **PDF files**: Vector format for printing and executive documents

### File Naming Convention
- `aifs_multimodal_architecture_diagram.{png,pdf}` - Complete system overview
- `aifs_cross_attention_detail.{png,pdf}` - Detailed attention mechanism

## Technical Details

### AIFS Architecture Features
- **Data Source**: ECMWF climate data (GRIB/NetCDF)
- **Model**: AIFS-Single-1.0 (MSE-trained)
- **Tokenizer**: TimeSeries transformer with 5D→token conversion
- **Integration**: Llama 3-8B language model
- **Fusion**: Multi-head cross-attention (32 heads)

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
