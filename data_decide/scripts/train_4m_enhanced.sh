#!/bin/bash
# Enhanced training script for 4M OLMo model with W&B and rich progress

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
export CUDA_VISIBLE_DEVICES="0"  # Use single GPU for 4M model

# Load environment variables (including W&B API key)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create output directories
mkdir -p checkpoints/olmo_4m_datadecide
mkdir -p logs/olmo_4m_datadecide

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     OLMo 4M Training with DataDecide Methodology           ║"
echo "║                                                            ║"
echo "║  • Dataset: 400M tokens from arXiv                         ║"
echo "║  • Model: 3.7M parameters (4M config)                      ║"
echo "║  • Training steps: 5,725                                   ║"
echo "║  • Learning rate: 1.4e-02                                  ║"
echo "║  • Weights & Biases: Enabled                               ║"
echo "║  • Rich progress indicators: Enabled                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if W&B is configured
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not found in environment"
    echo "Run './setup_wandb.py' to configure Weights & Biases"
    echo ""
    read -p "Continue without W&B? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run enhanced training
echo "Starting enhanced training..."
python train_enhanced.py 2>&1 | tee logs/olmo_4m_datadecide/training_enhanced.log

echo ""
echo "Training completed! Check logs/olmo_4m_datadecide/training_enhanced.log for details."
echo ""
echo "To monitor training in real-time, run in another terminal:"
echo "  python monitor_training.py"
echo ""
echo "To analyze results after training:"
echo "  python analyze_run.py <wandb_run_path>"