#!/bin/bash
#SBATCH --job-name=sft-training
#SBATCH --partition=blancapeak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sft_training_%j.out
#SBATCH --error=logs/sft_training_%j.err

# ============================================================================
# SLURM Job Script for Supervised Fine-Tuning (SFT)
# ============================================================================
# This script runs multi-GPU SFT training, typically after RL pre-training.
# SFT usually requires fewer resources than RL training.
#
# Resources requested:
#   - 1 node with 4 GPUs (NVIDIA GH200 120GB)
#   - 72 CPUs per task for data loading
#   - 800GB memory
#
# Usage:
#   sbatch slurm_sft_training.sh [OPTIONS]
#
# To run after RL training:
#   sbatch --dependency=afterok:<rl_job_id> slurm_sft_training.sh
# ============================================================================

set -euo pipefail

# Print job info
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-4}"
echo "Node list: ${SLURM_NODELIST}"
echo "============================================"

# ============================================================================
# Configuration
# ============================================================================

# Paths - modify these according to your setup
PROJECT_DIR="${PROJECT_DIR:-/lus/scratch/arigazzi/HPE-LLM4Climate}"
ZARR_DATA="${ZARR_DATA:-${PROJECT_DIR}/data/real_ecmwf_latest.zarr}"

# Checkpoint directory - can be from a previous RL run
# Set RL_CHECKPOINT_DIR to load from RL training, otherwise uses fresh model
RL_CHECKPOINT_DIR="${RL_CHECKPOINT_DIR:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints/sft_training_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft_model_${SLURM_JOB_ID}}"

# If RL checkpoint is specified, use it as the checkpoint dir for continuity
if [[ -n "${RL_CHECKPOINT_DIR}" ]]; then
    CHECKPOINT_DIR="${RL_CHECKPOINT_DIR}"
    echo "Using RL checkpoint from: ${RL_CHECKPOINT_DIR}"
fi

# Model configuration
MODEL_NAME="${MODEL_NAME:-mistralai/Ministral-3-8B-Instruct-2512}"

# Training hyperparameters
SFT_EPOCHS="${SFT_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SEED="${SEED:-42}"

# ============================================================================
# Environment Setup
# ============================================================================

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

# Create checkpoint and output directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Load modules if needed (adjust for your cluster)
# module load cuda/12.1
# module load python/3.11

# Activate virtual environment if exists
if [[ -f "${PROJECT_DIR}/venv/bin/activate" ]]; then
    source "${PROJECT_DIR}/venv/bin/activate"
fi

# Change to project directory
cd "${PROJECT_DIR}"

# Set Hugging Face token for model downloads (optional but recommended)
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN="${HF_TOKEN:-}"

# Set shared HF cache directory on the shared filesystem
# This ensures all nodes use the same cache, avoiding download race conditions
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}"

# ============================================================================
# Distributed Training Setup
# ============================================================================

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29501}"

# Calculate world size
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
# Handle format like "gpu:nvidia_gh200_120gb:4(S:0-3)" by extracting the number
if [[ "${GPUS_PER_NODE}" =~ :([0-9]+) ]]; then
    GPUS_PER_NODE="${BASH_REMATCH[1]}"
fi
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "============================================"
echo "Distributed Training Configuration"
echo "Master Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "World Size: ${WORLD_SIZE}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "============================================"

# Export distributed environment variables
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE

# NCCL settings for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============================================================================
# Launch Training
# ============================================================================

echo "Starting SFT training at $(date)"
echo "============================================"

# Use srun to launch distributed training
# Each task gets one GPU
srun --ntasks="${SLURM_NTASKS}" \
     --gpus-per-task=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     bash -c '
        # Set local rank based on SLURM local task ID
        export LOCAL_RANK=${SLURM_LOCALID}
        export RANK=${SLURM_PROCID}

        # Make all GPUs visible so that LOCAL_RANK device mapping works correctly
        export CUDA_VISIBLE_DEVICES=0,1,2,3

        echo "Node: $(hostname), Rank: ${RANK}, Local Rank: ${LOCAL_RANK}, GPU: ${CUDA_VISIBLE_DEVICES}"

        python -m multimodal_aifs.training.train_pipeline \
            --stage sft \
            --model-name "'"${MODEL_NAME}"'" \
            --zarr-paths "'"${ZARR_DATA}"'" \
            --checkpoint-dir "'"${CHECKPOINT_DIR}"'" \
            --output-dir "'"${OUTPUT_DIR}"'" \
            --sft-epochs "'"${SFT_EPOCHS}"'" \
            --batch-size "'"${BATCH_SIZE}"'" \
            --learning-rate "'"${LEARNING_RATE}"'" \
            --seed "'"${SEED}"'" \
            --device cuda
     '

TRAIN_EXIT_CODE=$?

echo "============================================"
echo "Training completed at $(date)"
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "Output model saved to: ${OUTPUT_DIR}"
echo "============================================"

exit ${TRAIN_EXIT_CODE}
