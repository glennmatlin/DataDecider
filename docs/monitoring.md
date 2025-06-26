# WANDB Monitoring Guide

This guide explains how to use the comprehensive WANDB monitoring system in DataDecider for tracking OLMo model training.

## Overview

DataDecider includes a complete telemetry system that provides:
- **Rich CLI Progress Display**: Beautiful terminal UI with progress bars and real-time metrics
- **WANDB Integration**: Cloud-based experiment tracking with comprehensive logging
- **System Monitoring**: GPU/CPU/memory usage tracking
- **Training Metrics**: Loss, learning rate, gradient statistics, and model performance

## Quick Start

### 1. Setup WANDB Credentials

Create a `.env` file in the project root:

```bash
# WANDB Configuration
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=finpile_datadecide
WANDB_ENTITY=your_username
WANDB_BASE_URL=https://api.wandb.ai

# DataDecider Configuration
DATADECIDER_LOG_LEVEL=INFO
```

**Security Note**: The `.env` file is already in `.gitignore` and will not be committed to version control.

### 2. Run Training with Monitoring

```bash
uv run python examples/train_olmo_pretokenized.py \
    --dataset synthetic_demo \
    --max-steps 1000 \
    --eval-steps 100 \
    --batch-size 8 \
    --gradient-accumulation 2 \
    --learning-rate 5e-4
```

This will automatically:
- Load WANDB credentials from environment variables
- Create a new WANDB run
- Display rich progress in the terminal
- Track all metrics to the cloud

### 3. Disable WANDB (Optional)

To run without WANDB tracking:

```bash
uv run python examples/train_olmo_pretokenized.py \
    --dataset synthetic_demo \
    --no-wandb
```

## Components

### TrainingMonitor

The central component that combines CLI progress display with WANDB tracking:

```python
from data_decide.utils.training_monitor import create_training_monitor

monitor = create_training_monitor(
    config={
        "model_size": "4m",
        "dataset": "synthetic_demo",
        "batch_size": 8,
        "learning_rate": 5e-4,
    },
    use_wandb=True,
    wandb_project="finpile_datadecide",  # Optional, uses env var by default
    quiet=False,
    verbose=False,
)

with monitor:
    # Your training code here
    monitor.log_model_info(model)
    monitor.start_training(training_config=args)
    # ... training loop ...
    monitor.log_metrics({"loss": loss, "lr": lr})
```

### ProgressManager

Provides rich terminal UI with progress bars:

```python
from data_decide.utils.progress_manager import ProgressManager

progress = ProgressManager(quiet=False, verbose=True)
with progress:
    progress.set_phase("training")
    progress.update_epoch(epoch, total_epochs)
    progress.update_step(step, total_steps, loss=loss)
```

Features:
- Multiple progress bars (epochs, steps, evaluation)
- Real-time metrics display
- System information panel
- Colored status indicators
- Time estimates and elapsed time

### WANDBManager

Handles all WANDB integration:

```python
from data_decide.utils.wandb_manager import WANDBManager

wandb_manager = WANDBManager(
    project="finpile_datadecide",  # Uses WANDB_PROJECT env var by default
    entity="your_username",       # Uses WANDB_ENTITY env var by default
    config={"model_size": "4m"},
    tags=["olmo", "experiment"],
)

with wandb_manager:
    wandb_manager.log_metrics({"loss": 1.5, "lr": 5e-4})
    wandb_manager.log_gradients(model)
    wandb_manager.log_checkpoint("./checkpoints/model.pt")
```

## What Gets Tracked

### System Metrics (Automatic)
- **GPU**: Utilization, memory usage, temperature
- **CPU**: Usage percentage
- **Memory**: System memory usage
- **Environment**: Python version, PyTorch version, CUDA info

### Training Metrics
- **Loss**: Training and validation loss
- **Learning Rate**: Per parameter group
- **Gradients**: Mean, std, min, max for each layer
- **Performance**: Samples per second, training time
- **Model**: Parameter count, model size

### Custom Metrics
- **Dataset Statistics**: Token counts, sequence lengths
- **Model Predictions**: Sample inputs/outputs (for text generation)
- **Checkpoints**: Model artifacts and metadata

## Configuration

### Telemetry Config File

Create `configs/telemetry_config.yaml` to customize monitoring:

```yaml
progress:
  enabled: true
  quiet: false
  verbose: false
  update_frequency: 1  # Log every N steps

wandb:
  enabled: true
  mode: "online"  # online, offline, disabled
  tags: ["olmo", "datadecider"]
  log_gradients: true
  log_predictions: true
  prediction_samples: 10

system_monitoring:
  enabled: true
  gpu_monitoring: true
  memory_monitoring: true
  update_frequency: 10  # Log every N steps

checkpoints:
  save_artifacts: true
  log_configs: true
```

### Environment Variables

All WANDB settings can be controlled via environment variables:

```bash
# Core settings
WANDB_API_KEY=your_api_key
WANDB_PROJECT=finpile_datadecide
WANDB_ENTITY=your_username
WANDB_BASE_URL=https://api.wandb.ai

# Optional settings
WANDB_MODE=online           # online, offline, disabled
WANDB_DIR=./wandb          # Local storage directory
WANDB_SILENT=false         # Suppress WANDB output
WANDB_CONSOLE=auto         # Console logging level
```

## Advanced Usage

### Custom Callbacks

For HuggingFace Transformers integration:

```python
from data_decide.utils.training_monitor import TrainingMonitor

class WANDBCallback(TrainerCallback):
    def __init__(self, monitor):
        self.monitor = monitor

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.monitor.log_metrics(logs, step=state.global_step)

# Add to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[WANDBCallback(monitor)],
    # ...
)
```

### Logging Predictions

```python
# Log sample predictions during training
predictions = model.generate(input_ids, max_length=50)
monitor.wandb.log_model_predictions(
    inputs=["Input text 1", "Input text 2"],
    predictions=["Generated text 1", "Generated text 2"],
    step=global_step,
    num_samples=5
)
```

### Logging Custom Visualizations

```python
import wandb

# Log learning rate schedule
steps = list(range(1000))
lrs = [lr_schedule(step) for step in steps]
monitor.wandb.run.log({
    "lr_schedule": wandb.plot.line_series(
        xs=steps,
        ys=[lrs],
        keys=["learning_rate"],
        title="Learning Rate Schedule",
        xname="Step"
    )
})
```

## Troubleshooting

### Common Issues

1. **"WANDB not found"**
   ```bash
   uv pip install wandb
   ```

2. **"Invalid API key"**
   - Check your `.env` file
   - Verify API key at https://wandb.ai/settings

3. **"Project not found"**
   - Ensure `WANDB_PROJECT` in `.env` matches your WANDB project
   - Create project in WANDB dashboard first

4. **"Base URL error"**
   - Use `https://api.wandb.ai` not `https://wandb.ai`

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("data_decide.utils.wandb_manager").setLevel(logging.DEBUG)
```

### Offline Mode

For environments without internet:

```bash
WANDB_MODE=offline uv run python examples/train_olmo_pretokenized.py
```

Sync later with:
```bash
wandb sync ./wandb/offline-run-*/
```

## Performance Impact

The monitoring system is designed to be lightweight:
- **System metrics**: ~0.1% overhead
- **Progress display**: Minimal impact
- **WANDB logging**: ~0.5% overhead
- **Total overhead**: <1% of training time

Disable heavy features if needed:
```python
monitor = create_training_monitor(
    use_wandb=True,
    config={"log_gradients": False, "log_predictions": False}
)
```

## Examples

See complete examples in:
- `examples/train_olmo_pretokenized.py` - Full training with monitoring
- `examples/train_olmo_with_telemetry.py` - Advanced telemetry features
- `configs/telemetry_config.yaml` - Configuration options

## Best Practices

1. **Use descriptive run names**:
   ```python
   wandb_name = f"olmo-{model_size}-{dataset}-{timestamp}"
   ```

2. **Tag experiments appropriately**:
   ```python
   tags = ["olmo", "4m", "synthetic", "ablation"]
   ```

3. **Log key hyperparameters**:
   ```python
   config = {
       "model_size": "4m",
       "batch_size": 8,
       "learning_rate": 5e-4,
       "dataset": "synthetic_demo",
   }
   ```

4. **Use groups for related experiments**:
   ```python
   wandb_group = "olmo-4m-ablation-study"
   ```

5. **Save important artifacts**:
   ```python
   monitor.wandb.log_checkpoint("./best_model.pt")
   monitor.wandb.log_config_file("./config.yaml")
   ```

This monitoring system provides comprehensive visibility into your OLMo training runs, making it easy to track progress, debug issues, and compare experiments.
