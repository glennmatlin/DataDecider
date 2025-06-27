# DataDecider

A framework for training and evaluating Open Language Models (OLMo) using the DataDecide methodology for efficient data curation and model development.

## Overview

DataDecider implements the DataDecide approach for training language models, which uses small-scale proxy experiments to predict which data mixtures will perform best at scale. This package provides:

- Complete OLMo model implementation with 14 size variants (4M to 1B parameters)
- DataDecide data curation pipeline with proxy metrics
- Training infrastructure with distributed support
- Evaluation suite for model assessment
- Integration with Weights & Biases for experiment tracking

## Installation

### From GitHub (for use in other projects)

```bash
# Using pip
pip install git+https://github.com/yourusername/DataDecider.git

# Using uv
uv pip install git+https://github.com/yourusername/DataDecider.git

# For local development from another project
pip install -e /path/to/DataDecider
```

### For Development

```bash
# Clone the repository
git clone https://github.com/yourusername/DataDecider.git
cd DataDecider

# Install in development mode with uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Your Dataset

The framework expects tokenized datasets in HuggingFace format. You can use the provided scripts to prepare your data:

```bash
# Build a dataset from raw files
python -m data_decide.scripts.prepare_training_data \
    --input-dir ./raw_data \
    --output-dir ./processed_data \
    --tokenizer EleutherAI/gpt-neox-20b \
    --max-length 2048
```

### 2. Configure Your Model

Configuration files are in YAML format. Example for 4M model:

```yaml
# configs/model_configs/olmo_4m.yaml
model_size: "4M"
model_params:
  num_layers: 8
  hidden_size: 64
  num_attention_heads: 8
  vocab_size: 50254
```

### 3. Train Your Model

```bash
# Using the main training script
data-decide-train \
    --config configs/training_configs/olmo_4m_training.yaml

# Or use the enhanced version with rich UI
python -m data_decide.scripts.train_enhanced \
    --config configs/training_configs/olmo_4m_training.yaml
```

### 4. Monitor Training

```bash
# Real-time monitoring with rich terminal UI
data-decide-monitor --run-name my_training_run

# Or analyze completed runs
data-decide-analyze --wandb-run-path username/project/run_id
```

## Key Features

### DataDecide Methodology

The DataDecide approach involves:

1. **Proxy Dataset Creation**: Generate multiple small datasets with different data mixtures
2. **Proxy Metrics**: Compute perplexity, diversity, and quality scores without full training
3. **Mixture Selection**: Choose the best data mixture based on proxy results
4. **Full Training**: Train the model on the selected data mixture

### Model Sizes

Supported OLMo configurations:

| Model Size | Parameters | Hidden Size | Layers | Heads |
|------------|------------|-------------|---------|--------|
| 4M         | 3.7M       | 64          | 8       | 8      |
| 20M        | 18.6M      | 128         | 16      | 16     |
| 38M        | 36.9M      | 192         | 16      | 16     |
| 70M        | 66.8M      | 256         | 18      | 16     |
| 160M       | 152.2M     | 384         | 20      | 16     |
| 410M       | 390.2M     | 640         | 24      | 16     |
| 1B         | 982.3M     | 1024        | 28      | 16     |

### Training Features

- **Distributed Training**: Full support for multi-GPU training via Accelerate
- **Mixed Precision**: FP16/BF16 training for efficiency
- **Gradient Checkpointing**: Memory-efficient training for larger models
- **Learning Rate Scheduling**: Cosine decay with warmup
- **Comprehensive Monitoring**: WANDB integration with system metrics and rich terminal UI
- **Pre-tokenized Data Pipeline**: Efficient training with separated tokenization

## Monitoring & Visualization

DataDecider includes a comprehensive monitoring system that provides both local and cloud-based tracking:

### Rich Terminal UI
- Real-time progress bars for epochs, steps, and evaluation
- Live metrics display (loss, learning rate, GPU usage)
- Beautiful colored output with system information
- Time estimates and performance metrics

### WANDB Integration
- Automatic experiment tracking to Weights & Biases
- System monitoring (GPU utilization, memory, temperature)
- Model metrics (gradients, learning rates, predictions)
- Checkpoint artifact management
- Hyperparameter tracking and visualization

### Quick Setup
```bash
# 1. Add to .env file
WANDB_API_KEY=your_api_key
WANDB_PROJECT=finpile_datadecide
WANDB_ENTITY=your_username

# 2. Run training (monitoring enabled by default)
uv run python examples/train_olmo_pretokenized.py --dataset tiny_100k
```

See [`docs/monitoring.md`](docs/monitoring.md) for complete documentation and [`docs/wandb-quickstart.md`](docs/wandb-quickstart.md) for a quick start guide.

## Project Structure

```
DataDecider/
├── configs/              # Configuration files
│   ├── model_configs/    # Model architecture configs
│   ├── training_configs/ # Training hyperparameters
│   └── data_configs/     # Data processing configs
├── data_decide/          # Main package
│   ├── olmo/            # OLMo implementation
│   │   ├── models/      # Model architecture
│   │   ├── data/        # Data processing
│   │   ├── training/    # Training logic
│   │   ├── evaluation/  # Evaluation metrics
│   │   └── utils/       # Utilities
│   └── scripts/         # Executable scripts
├── tests/               # Unit tests
└── data/               # Data directory (gitignored)
```

## Data Management

This repository does not include the large training datasets. To obtain the data:

1. **Sample Data**: A small sample dataset is included in `tests/test_data/` for testing
2. **Full Datasets**: See `data/README.md` for instructions on downloading the full arXiv datasets
3. **Custom Data**: Use the data preparation scripts to process your own datasets

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Weights & Biases
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=olmo-datadecide
WANDB_ENTITY=your_entity

# Training
CUDA_VISIBLE_DEVICES=0,1,2,3
TOKENIZERS_PARALLELISM=false
```

### Training Configuration

Example training configuration:

```yaml
# Training parameters
model_size: "4M"
data_path: "./data/processed/olmo_4m_400M_tokens"
output_dir: "./checkpoints/olmo_4m_datadecide"
num_train_epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 1.4e-2
warmup_steps: 572
save_steps: 1000
eval_steps: 500
logging_steps: 10

# W&B configuration
report_to: ["wandb"]
wandb_project: "olmo-4m-datadecide"
wandb_name: "olmo-4m-arxiv-400M"
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=data_decide

# Run specific test
pytest tests/test_data_curation.py
```

### Code Quality

```bash
# Format code
ruff format .

# Check style
ruff check .
```

## Using DataDecider in Your Project

To use DataDecider in another project (like FinPileCode):

```python
from data_decide.olmo.models import OLMoForCausalLM, OLMoConfig
from data_decide.olmo.data import DataDecideCurator

# Initialize model
config = OLMoConfig.from_pretrained("olmo-4m")
model = OLMoForCausalLM(config)

# Use DataDecide for data curation
curator = DataDecideCurator()
proxy_datasets = curator.create_proxy_datasets(your_data)
best_mixture = curator.select_best_mixture(proxy_datasets)
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{datadecider,
  title = {DataDecider: OLMo Training with DataDecide Methodology},
  author = {FinPile Team},
  year = {2024},
  url = {https://github.com/yourusername/DataDecider}
}
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgments

- OLMo architecture based on the paper "OLMo: Accelerating the Science of Language Models"
- DataDecide methodology for efficient data curation
- Built with HuggingFace Transformers and Accelerate
