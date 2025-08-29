# Documentation Directory

This directory contains comprehensive documentation and visual materials for the HPE LLM4Climate multimodal architecture.

## 📋 Architecture Documentation

### **Architecture Materials**
- **`architecture_documentation.md`** - Comprehensive technical documentation
- **`multimodal_architecture_diagram.pdf`** - Main system architecture diagram
- **`cross_attention_detail.pdf`** - Detailed cross-attention mechanism diagram

### **Presentation Materials**
- **`multimodal_architecture_diagram.png`** - High-resolution main diagram for slides
- **`cross_attention_detail.png`** - High-resolution attention detail for technical presentations

### **Source Code**
- **`create_architecture_diagram.py`** - Source code for generating main architecture diagram
- **`create_attention_detail.py`** - Source code for generating detailed attention mechanism diagram

## 🎯 Usage

### For Formal Presentations
Use the PDF versions for formal presentations and architecture reviews:
```
docs/multimodal_architecture_diagram.pdf
docs/cross_attention_detail.pdf
```

### For Technical Slides
Use the PNG versions for technical presentations and documentation:
```
docs/multimodal_architecture_diagram.png
docs/cross_attention_detail.png
```

### For Documentation
Reference the comprehensive technical documentation:
```
docs/architecture_documentation.md
```

### To Regenerate Diagrams
Run the source scripts to update diagrams with any architectural changes:
```bash
# Generate main architecture diagram
python docs/create_architecture_diagram.py

# Generate detailed attention mechanism diagram
python docs/create_attention_detail.py
```

## 📊 Diagram Contents

### Main Architecture Diagram
- Complete multimodal fusion pipeline
- Input specifications and tensor dimensions
- Preprocessing steps (normalization, tokenization, location processing)
- Cross-attention fusion mechanism
- Technical specifications and implementation details

### Cross-Attention Detail Diagram
- Mathematical formulation of attention mechanism
- Multi-head attention computation flow
- Tensor dimensions at each processing step
- Technical implementation parameters
- Attention pattern visualization
- Linear projection and head splitting details

## 🔄 Updating Documentation

When making architectural changes:

1. **Update source code**: Modify the Python diagram generation scripts
2. **Regenerate diagrams**: Run the scripts to create updated visuals
3. **Update documentation**: Revise `architecture_documentation.md` with changes
4. **Commit changes**: Include both source and generated files in version control

## 📁 File Organization

```
docs/
├── README.md                              # This file
├── architecture_documentation.md          # Comprehensive technical documentation
├── multimodal_architecture_diagram.pdf    # Main diagram (PDF)
├── multimodal_architecture_diagram.png    # Main diagram (PNG)
├── cross_attention_detail.pdf             # Attention detail (PDF)
├── cross_attention_detail.png             # Attention detail (PNG)
├── create_architecture_diagram.py         # Main diagram generator
└── create_attention_detail.py             # Attention detail generator
```

## 🎯 Best Practices

- **PDF for presentations**: Use PDF versions for formal presentations and reviews
- **PNG for slides**: Use PNG versions for technical slides and web documentation
- **Source control**: Keep both source scripts and generated files in version control
- **Regular updates**: Regenerate diagrams when architecture changes
- **Documentation sync**: Keep written documentation aligned with visual diagrams
