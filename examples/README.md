# Training Examples

This directory contains working training scripts and model implementations migrated from FinPileCode.

## Files

### train_olmo_gpu.py
Working GPU training script that successfully trains a language model with perplexity evaluation.

**Features**:
- GPU optimization with FP16
- PerplexityEvaluator class
- Fallback to GPT-NeoX when OLMo fails
- Comprehensive logging

**Results achieved**:
- Model: 6.8M parameters
- Validation perplexity: 8,882
- GPU memory: 1.39 GB

### olmo_wrapper.py
OLMo-compatible model using GPT-NeoX as the base architecture.

**Features**:
- SwiGLU activation (like OLMo)
- OLMo dimensions (64 hidden, 8 layers, 8 heads)
- Full compatibility with HuggingFace Trainer
- Easy model creation with `create_olmo_model()`

**Usage**:
```python
from olmo_wrapper import create_olmo_model

model = create_olmo_model("4M")  # Creates 3.7M parameter model
```

### train_olmo_wrapper_gpu.py
Training script specifically for the OLMo wrapper model.

**Features**:
- Higher batch size (32) with gradient accumulation
- Optimized hyperparameters
- Full perplexity evaluation
- Generation testing

**Configuration**:
- Effective batch size: 64
- Learning rate: 1.4e-2 (from OLMo paper)
- FP16 training
- 100 training steps

## Quick Start

1. **Test the wrapper model**:
   ```bash
   python olmo_wrapper.py
   ```

2. **Run GPU training**:
   ```bash
   python train_olmo_gpu.py
   ```

3. **Train with wrapper**:
   ```bash
   python train_olmo_wrapper_gpu.py
   ```

## Data Location

All scripts expect data at:
```
/home/gmatlin/Codespace/DataDecider/data/raw/arxiv_sample.json.gz
```

This file contains 100 arXiv papers for testing.

## Performance Tips

1. Use FP16 for memory efficiency
2. Start with batch size 8 for 8GB GPUs
3. Monitor GPU memory with `nvidia-smi`
4. Use gradient accumulation for larger effective batch sizes
