# OLMo Training Pipeline

This directory contains the implementation of the OLMo (Open Language Model) training pipeline with DataDecide methodology for efficient data curation and selection.

## Overview

The pipeline supports training models ranging from 4M to 1B parameters with systematic experimentation and evaluation protocols.

### Key Components

- **Data Curation** (`data/`): DataDecide methodology for optimal dataset selection
- **Model Architecture** (`models/`): OLMo model implementation with RoPE and SwiGLU
- **Training Infrastructure** (`training/`): Distributed training with mixed precision support
- **Evaluation Pipeline** (`evaluation/`): Comprehensive benchmark evaluation suite
- **Utilities** (`utils/`): Logging, checkpointing, and configuration management

## Model Sizes

The pipeline supports 14 model variants:
- 4M, 6M, 8M, 10M, 14M, 16M, 20M (Small-scale experiments)
- 60M, 90M, 150M (Medium-scale models)
- 300M, 530M, 750M, 1B (Large-scale models)

## Quick Start

### Training a Model

```bash
# Single GPU training
python scripts/olmo/train.py \
    --model_size 150M \
    --data_path ./data \
    --output_dir ./outputs \
    --config_file configs/training_configs/default_training.yaml

# Multi-GPU distributed training
./scripts/olmo/launch_distributed.sh 150M 8 ./data ./outputs
```

### Configuration

- Model configs: `configs/model_configs/`
- Training configs: `configs/training_configs/`
- Data configs: `configs/data_configs/`

### DataDecide Methodology

The pipeline uses DataDecide for efficient data curation:
1. Creates small-scale proxy experiments
2. Evaluates different data sampling strategies
3. Selects optimal data composition before full training

## Architecture Features

- **Rotary Position Embeddings (RoPE)**
- **SwiGLU activation function**
- **Gradient checkpointing for memory efficiency**
- **Mixed precision training (FP16/BF16)**
- **Distributed training with Accelerate**

## Evaluation

The evaluation suite includes benchmarks:
- PIQA, HellaSwag, WinoGrande
- ARC (Easy & Challenge)
- BoolQ, OpenBookQA
- MMLU, TruthfulQA, GSM8K