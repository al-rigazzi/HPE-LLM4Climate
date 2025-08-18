#!/bin/bash
# Environment setup script for multimodal climate-text training
# This script helps set up the training environment with all necessary dependencies.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🌍 HPE LLM4Climate Training Environment Setup${NC}"
echo "=================================================="

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${GREEN}✓ Python version is compatible${NC}"
else
    echo -e "${RED}✗ Python 3.8+ required${NC}"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓ Virtual environment detected: $VIRTUAL_ENV${NC}"
elif [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo -e "${GREEN}✓ Conda environment detected: $CONDA_DEFAULT_ENV${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment detected. Consider using venv or conda.${NC}"
fi

# Function to install requirements
install_requirements() {
    local req_file=$1
    local description=$2

    echo -e "${YELLOW}Installing $description...${NC}"

    if [ ! -f "$req_file" ]; then
        echo -e "${RED}✗ Requirements file not found: $req_file${NC}"
        return 1
    fi

    pip install -r "$req_file"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $description installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install $description${NC}"
        return 1
    fi
}

# Parse command line arguments
CUDA_VERSION=""
FORCE_REINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda VERSION  Specify CUDA version (e.g., cu118, cu121)"
            echo "  --force         Force reinstall of packages"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Install requirements"
            echo "  $0 --cuda cu118       # Install with CUDA 11.8 support"
            echo "  $0 --force            # Force reinstall packages"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch with appropriate CUDA support
echo -e "${YELLOW}Installing PyTorch...${NC}"
if [ ! -z "$CUDA_VERSION" ]; then
    echo "Installing PyTorch with CUDA $CUDA_VERSION support"
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
elif command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
install_requirements "requirements.txt" "requirements"

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"

# Check core packages
packages=("torch" "transformers" "deepspeed" "numpy" "yaml")
for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package${NC}"
    else
        echo -e "${RED}✗ $package${NC}"
    fi
done

# Check CUDA availability
echo -e "${YELLOW}Checking CUDA availability...${NC}"
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$cuda_available" = "True" ]; then
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓ CUDA available with $gpu_count GPU(s)${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not available (CPU training only)${NC}"
fi

# Check DeepSpeed
echo -e "${YELLOW}Checking DeepSpeed...${NC}"
if python -c "import deepspeed" 2>/dev/null; then
    echo -e "${GREEN}✓ DeepSpeed available${NC}"
else
    echo -e "${YELLOW}⚠️  DeepSpeed not available${NC}"
    echo "You may need to install additional dependencies or compile from source"
fi

echo ""
echo -e "${GREEN}🎉 Environment setup completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Prepare your training data: python prepare_data.py"
echo "2. Configure training: edit config.yaml"
echo "3. Start training: ./launch.sh single-gpu"
echo ""
echo "For distributed training:"
echo "  ./launch.sh multi-gpu --num-gpus 4"
echo ""
echo "For help:"
echo "  python train_multimodal.py --help"
echo "  ./launch.sh --help"
