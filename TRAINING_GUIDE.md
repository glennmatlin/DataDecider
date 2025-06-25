# OLMo 4M Training Guide with DataDecide

This guide covers training the 4M OLMo model using the DataDecide methodology with enhanced monitoring.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Weights & Biases** (recommended):
   ```bash
   python setup_wandb.py
   ```
   This will guide you through setting up your W&B API key.

3. **Dataset**: The 400M token arXiv dataset should already be prepared at:
   ```
   data/processed/olmo_4m_400M_tokens/
   ```

## Training Options

### Option 1: Enhanced Training with Full Monitoring (Recommended)

Use the enhanced training script with W&B integration and rich progress indicators:

```bash
cd data_decide
./scripts/train_4m_enhanced.sh
```

This provides:
- Real-time progress bars
- W&B metric tracking
- DataDecide proxy metrics
- GPU memory monitoring
- Time estimates and ETA
- Automatic checkpointing

### Option 2: Standard Training with Existing Infrastructure

Use the original training script with DataDecide:

```bash
cd data_decide
./scripts/train_4m_datadecide.sh
```

### Option 3: Standalone Training (Fallback)

If you encounter build issues:

```bash
cd data_decide
python train_standalone.py
```

Or use the enhanced standalone version:

```bash
python train_enhanced.py
```

## Monitoring Training

### Live Monitoring Dashboard

In a separate terminal, run:

```bash
python monitor_training.py
```

This shows:
- Current metrics (loss, perplexity, learning rate)
- Training progress with ETA
- Best metrics achieved
- DataDecide scores
- Link to W&B dashboard

### W&B Dashboard

View your training online at:
- https://wandb.ai/your-username/olmo-4m-datadecide

The dashboard includes:
- Real-time loss curves
- Learning rate schedule
- DataDecide metrics evolution
- System metrics (GPU, memory)
- Hyperparameter tracking

## Post-Training Analysis

After training completes, analyze the run:

```bash
python analyze_run.py your-username/olmo-4m-datadecide/run-id
```

This generates:
- Training plots (`training_analysis.png`)
- DataDecide metrics plots (`datadecide_metrics.png`)
- Configuration export (`config.json`)
- Full history CSV (`history.csv`)
- Summary report (`report.txt`)

## Configuration

### Key Hyperparameters (from OLMo paper)

- **Model**: 3.7M parameters (called "4M")
- **Hidden size**: 64
- **Layers**: 8
- **Attention heads**: 8
- **Learning rate**: 1.4e-02
- **Batch size**: 32
- **Training steps**: 5,725
- **Warmup steps**: 57 (1%)
- **Sequence length**: 2048

### DataDecide Methodology

The training uses DataDecide proxy experiments:
1. Creates 5 different data recipes from the dataset
2. Evaluates each using entropy-based perplexity, diversity, and quality metrics
3. Selects the best recipe based on combined score
4. Tracks proxy metrics throughout training

### Checkpointing

Checkpoints are saved every 500 steps to:
```
checkpoints/olmo_4m_datadecide/checkpoint-{step}/
```

Final model saved to:
```
checkpoints/olmo_4m_datadecide/final_model/
```

## Troubleshooting

### W&B Issues

If W&B isn't working:
1. Check `.env` file exists with your API key
2. Run `wandb login` manually
3. Set `WANDB_MODE=offline` for offline mode

### Memory Issues

For GPU memory problems:
- Reduce batch size in `configs/training_configs/olmo_4m_training.yaml`
- Enable gradient checkpointing (though not needed for 4M model)

### Progress Not Showing

If rich progress bars don't display properly:
- Check terminal supports Unicode
- Try setting: `export TERM=xterm-256color`

## Expected Training Time

With default settings:
- **Total steps**: 5,725
- **Tokens/step**: 65,536 (32 batch × 2048 seq_len)
- **Total tokens**: 375M (will repeat dataset)
- **Estimated time**: 2-4 hours on a single GPU

## Next Steps

After training:
1. Evaluate on downstream tasks using `olmo/evaluation/evaluator.py`
2. Fine-tune for specific applications
3. Compare with larger model sizes
4. Use insights from DataDecide metrics to improve data curation