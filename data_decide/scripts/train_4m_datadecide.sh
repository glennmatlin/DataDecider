#!/bin/bash
# Train 4M OLMo model with DataDecide methodology

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
export CUDA_VISIBLE_DEVICES="0"  # Use single GPU for 4M model

# Create output directories
mkdir -p checkpoints/olmo_4m_datadecide
mkdir -p logs/olmo_4m_datadecide

echo "Starting OLMo 4M training with DataDecide methodology..."
echo "Dataset: 400M tokens from arXiv"
echo "Model: 3.7M parameters (4M config)"
echo "Training steps: 5,725"
echo "Learning rate: 1.4e-02"
echo ""

# Run training with DataDecide
python scripts/train.py \
    --model_size 4M \
    --data_path data/processed/olmo_4m_400M_tokens \
    --output_dir checkpoints/olmo_4m_datadecide \
    --config_file configs/training_configs/olmo_4m_training.yaml \
    --use_data_decide \
    2>&1 | tee logs/olmo_4m_datadecide/training.log

echo ""
echo "Training completed! Check logs/olmo_4m_datadecide/training.log for details."