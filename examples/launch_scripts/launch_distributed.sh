#!/bin/bash
# scripts/launch_distributed.sh

# Configuration
MODEL_SIZE=${1:-"150M"}
NUM_GPUS=${2:-8}
DATA_PATH=${3:-"./data"}
OUTPUT_DIR=${4:-"./outputs"}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Launch distributed training
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29500 \
    scripts/train.py \
    --model_size "$MODEL_SIZE" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --config_file configs/training_configs/default_training.yaml \
    --distributed
