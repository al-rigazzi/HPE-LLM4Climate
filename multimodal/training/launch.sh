#!/bin/bash
"""
Launch script for multimodal climate-text fusion training

This script provides convenient commands for different training scenarios.
"""

set -e

# Default configuration
CONFIG_FILE="config.yaml"
DEEPSPEED_CONFIG="deepspeed_config.json"
NUM_GPUS=""
NUM_NODES=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  single-gpu     Train on single GPU"
    echo "  multi-gpu      Train on multiple GPUs with DeepSpeed"
    echo "  distributed    Train across multiple nodes"
    echo "  prepare-data   Prepare dummy training data"
    echo "  validate-data  Validate data format"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE        Training config file (default: config.yaml)"
    echo "  -d, --deepspeed FILE     DeepSpeed config file (default: deepspeed_config.json)"
    echo "  -g, --num-gpus N         Number of GPUs to use"
    echo "  -n, --num-nodes N        Number of nodes for distributed training (default: 1)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 single-gpu"
    echo "  $0 multi-gpu --num-gpus 4"
    echo "  $0 distributed --num-gpus 8 --num-nodes 2"
    echo "  $0 prepare-data"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--deepspeed)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        -g|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -n|--num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        single-gpu|multi-gpu|distributed|prepare-data|validate-data)
            COMMAND="$1"
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "${COMMAND}" ]; then
    echo -e "${RED}Error: No command specified${NC}"
    usage
    exit 1
fi

# Utility functions
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File not found: $1${NC}"
        exit 1
    fi
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found${NC}"
        exit 1
    fi
    
    # Check required packages
    python -c "import torch; import transformers; import deepspeed" 2>/dev/null || {
        echo -e "${RED}Error: Missing required packages. Install with:${NC}"
        echo "pip install -r requirements-training.txt"
        exit 1
    }
    
    # Check GPU availability for GPU training
    if [ "$COMMAND" != "prepare-data" ] && [ "$COMMAND" != "validate-data" ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo -e "${YELLOW}Warning: CUDA not available, will use CPU training${NC}"
        fi
    fi
    
    echo -e "${GREEN}Dependencies check passed${NC}"
}

# Execute commands
case $COMMAND in
    single-gpu)
        echo -e "${GREEN}Starting single GPU training...${NC}"
        check_dependencies
        check_file "$CONFIG_FILE"
        
        echo "Config: $CONFIG_FILE"
        echo "Command: python train_multimodal.py --config $CONFIG_FILE"
        echo ""
        
        python train_multimodal.py --config "$CONFIG_FILE"
        ;;
        
    multi-gpu)
        echo -e "${GREEN}Starting multi-GPU training with DeepSpeed...${NC}"
        check_dependencies
        check_file "$CONFIG_FILE"
        check_file "$DEEPSPEED_CONFIG"
        
        # Auto-detect GPUs if not specified
        if [ -z "$NUM_GPUS" ]; then
            NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
            echo -e "${YELLOW}Auto-detected $NUM_GPUS GPUs${NC}"
        fi
        
        echo "Config: $CONFIG_FILE"
        echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
        echo "Number of GPUs: $NUM_GPUS"
        echo "Command: deepspeed --num_gpus=$NUM_GPUS train_multimodal.py --config $CONFIG_FILE --deepspeed $DEEPSPEED_CONFIG"
        echo ""
        
        deepspeed --num_gpus="$NUM_GPUS" train_multimodal.py \
            --config "$CONFIG_FILE" \
            --deepspeed "$DEEPSPEED_CONFIG"
        ;;
        
    distributed)
        echo -e "${GREEN}Starting distributed training across $NUM_NODES nodes...${NC}"
        check_dependencies
        check_file "$CONFIG_FILE"
        check_file "$DEEPSPEED_CONFIG"
        
        if [ -z "$NUM_GPUS" ]; then
            echo -e "${RED}Error: --num-gpus must be specified for distributed training${NC}"
            exit 1
        fi
        
        echo "Config: $CONFIG_FILE"
        echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
        echo "Number of GPUs: $NUM_GPUS"
        echo "Number of Nodes: $NUM_NODES"
        echo "Command: deepspeed --num_gpus=$NUM_GPUS --num_nodes=$NUM_NODES train_multimodal.py --config $CONFIG_FILE --deepspeed $DEEPSPEED_CONFIG"
        echo ""
        
        deepspeed --num_gpus="$NUM_GPUS" --num_nodes="$NUM_NODES" train_multimodal.py \
            --config "$CONFIG_FILE" \
            --deepspeed "$DEEPSPEED_CONFIG"
        ;;
        
    prepare-data)
        echo -e "${GREEN}Preparing training data...${NC}"
        
        # Check if data directory exists
        DATA_DIR="data/training"
        if [ -d "$DATA_DIR" ]; then
            echo -e "${YELLOW}Warning: Data directory $DATA_DIR already exists${NC}"
            read -p "Do you want to overwrite it? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Cancelled."
                exit 0
            fi
            rm -rf "$DATA_DIR"
        fi
        
        echo "Creating dummy data in $DATA_DIR..."
        python prepare_data.py --output_dir "$DATA_DIR" --num_samples 1000
        
        echo -e "${GREEN}Data preparation completed!${NC}"
        echo "Training samples: $DATA_DIR/train/"
        echo "Validation samples: $DATA_DIR/val/"
        ;;
        
    validate-data)
        echo -e "${GREEN}Validating data format...${NC}"
        
        DATA_DIR="data/training"
        if [ ! -d "$DATA_DIR" ]; then
            echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
            echo "Run '$0 prepare-data' to create dummy data, or specify correct path in config.yaml"
            exit 1
        fi
        
        python prepare_data.py --output_dir "$DATA_DIR" --validate_only
        ;;
        
    *)
        echo -e "${RED}Error: Unknown command: $COMMAND${NC}"
        usage
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"
