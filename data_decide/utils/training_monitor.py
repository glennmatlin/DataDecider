"""Combined training monitor for DataDecider with progress display and WANDB tracking."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from ..utils.logging_utils import get_logger
from .progress_manager import ProgressManager
from .wandb_manager import WANDBManager

logger = get_logger(__name__)


class TrainingMonitor:
    """Unified training monitor combining CLI progress and WANDB tracking."""

    def __init__(
        self,
        # Progress manager settings
        quiet: bool = False,
        verbose: bool = False,
        # WANDB settings
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_mode: str = "online",
        wandb_dir: Optional[str] = None,
        wandb_group: Optional[str] = None,
        # General settings
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./outputs",
    ):
        """Initialize training monitor.

        Args:
            quiet: Suppress CLI output
            verbose: Show detailed output
            use_wandb: Enable WANDB tracking
            wandb_project: WANDB project name
            wandb_entity: WANDB entity
            wandb_name: Run name
            wandb_tags: Run tags
            wandb_mode: WANDB mode (online/offline/disabled)
            wandb_dir: WANDB directory
            wandb_group: WANDB group
            config: Training configuration
            output_dir: Output directory for logs
        """
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress manager
        self.progress = ProgressManager(quiet=quiet, verbose=verbose)

        # Initialize WANDB manager
        self.use_wandb = use_wandb and wandb_mode != "disabled"
        self.wandb = None

        if self.use_wandb:
            try:
                self.wandb = WANDBManager(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config,
                    tags=wandb_tags,
                    name=wandb_name,
                    mode=wandb_mode,
                    dir=wandb_dir or str(self.output_dir / "wandb"),
                    group=wandb_group,
                    job_type="train",
                )
            except Exception as e:
                logger.warning(f"Failed to initialize WANDB: {e}")
                self.use_wandb = False

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.metrics_history = []

        # Timing
        self.epoch_start_time = None
        self.step_start_time = None
        self.training_start_time = None

    def start_training(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        num_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
    ):
        """Start training session.

        Args:
            model_config: Model configuration
            training_config: Training configuration
            num_epochs: Total number of epochs
            total_steps: Total number of steps
        """
        self.training_start_time = time.time()

        # Update configuration
        if model_config:
            self.config["model_config"] = model_config
        if training_config:
            self.config["training_config"] = training_config

        # Set progress totals
        if num_epochs:
            self.progress.update_epoch(0, num_epochs)
        if total_steps:
            self.progress.update_step(0, total_steps)

        # Initialize WANDB run
        if self.use_wandb and self.wandb.run is None:
            self.wandb.init_run()

        # Display configuration
        self.progress.set_phase("initializing")
        self.progress.print_config(self.config, "Training Configuration")

        # Log system info
        system_info = self._get_system_info()
        self.progress.print_system_info(system_info)

        self.progress.log_message("Training started", "success")

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        info = {
            "PyTorch Version": torch.__version__,
            "CUDA Available": str(torch.cuda.is_available()),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "GPU Device": torch.cuda.get_device_name(0),
                    "GPU Memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                    "CUDA Version": torch.version.cuda,
                }
            )

        return info

    def start_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        """Start a new epoch.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        self.epoch = epoch
        self.epoch_start_time = time.time()

        self.progress.set_phase("training")
        if total_epochs:
            self.progress.update_epoch(epoch, total_epochs)

        self.progress.log_message(f"Starting epoch {epoch}", "info")

    def end_epoch(self, metrics: Optional[Dict[str, float]] = None):
        """End current epoch.

        Args:
            metrics: Epoch metrics to log
        """
        epoch_time = time.time() - self.epoch_start_time

        # Log epoch completion
        self.progress.log_message(f"Epoch {self.epoch} completed in {self.progress.format_time(epoch_time)}", "success")

        # Log metrics
        if metrics:
            metrics["epoch"] = self.epoch
            metrics["epoch_time"] = epoch_time
            self.log_metrics(metrics, prefix="epoch")

    def update_step(
        self,
        step: int,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Update training step.

        Args:
            step: Current step
            total_steps: Total steps
            loss: Current loss
            metrics: Additional metrics
        """
        self.global_step = step

        # Update progress bar
        if total_steps:
            self.progress.update_step(step, total_steps, loss)

        # Log metrics
        if metrics or loss is not None:
            step_metrics = metrics or {}
            if loss is not None:
                step_metrics["loss"] = loss
            self.log_metrics(step_metrics, step=step, prefix="train")

    def start_evaluation(self, num_batches: Optional[int] = None):
        """Start evaluation phase.

        Args:
            num_batches: Number of evaluation batches
        """
        self.progress.set_phase("evaluating")
        if num_batches:
            self.progress.update_eval(0, num_batches)

        self.progress.log_message("Starting evaluation", "info")

    def update_evaluation(self, current: int, total: int):
        """Update evaluation progress.

        Args:
            current: Current batch
            total: Total batches
        """
        self.progress.update_eval(current, total)

    def end_evaluation(self, metrics: Dict[str, float]):
        """End evaluation phase.

        Args:
            metrics: Evaluation metrics
        """
        # Log metrics
        self.log_metrics(metrics, step=self.global_step, prefix="eval")

        # Check for best model
        metric_name = "loss"
        if metric_name in metrics and metrics[metric_name] < self.best_metric:
            self.best_metric = metrics[metric_name]
            self.progress.log_message(f"New best model! {metric_name}: {self.best_metric:.4f}", "success")

            # Save best metric
            best_metric_path = self.output_dir / "best_metric.json"
            import json

            with open(best_metric_path, "w") as f:
                json.dump(
                    {
                        "metric": metric_name,
                        "value": self.best_metric,
                        "step": self.global_step,
                        "epoch": self.epoch,
                    },
                    f,
                    indent=2,
                )

        # Display metrics
        self.progress.update_metrics(metrics)
        self.progress.log_message("Evaluation completed", "success")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        """Log metrics to all backends.

        Args:
            metrics: Metrics to log
            step: Step number
            prefix: Prefix for metric names
        """
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Update progress display
        self.progress.update_metrics(metrics)

        # Log to WANDB
        if self.use_wandb and self.wandb:
            self.wandb.log_metrics(metrics, step=step)

        # Store in history
        metrics_record = metrics.copy()
        metrics_record["step"] = step or self.global_step
        metrics_record["timestamp"] = time.time()
        self.metrics_history.append(metrics_record)

    def log_model_info(self, model: torch.nn.Module):
        """Log model information.

        Args:
            model: PyTorch model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

        self.progress.log_message(f"Model loaded: {trainable_params:,} trainable parameters", "success")

        # Log to WANDB
        if self.use_wandb and self.wandb and self.wandb.run:
            self.wandb.run.config.update({"model": model_info})

    def log_learning_rate(self, optimizer: torch.optim.Optimizer):
        """Log current learning rate.

        Args:
            optimizer: PyTorch optimizer
        """
        if self.use_wandb and self.wandb:
            self.wandb.log_learning_rate(optimizer, step=self.global_step)

    def log_gradients(self, model: torch.nn.Module):
        """Log model gradients.

        Args:
            model: PyTorch model
        """
        if self.use_wandb and self.wandb:
            self.wandb.log_gradients(model, step=self.global_step)

    def log_predictions(
        self,
        inputs: List[str],
        predictions: List[str],
        targets: Optional[List[str]] = None,
        num_samples: int = 5,
    ):
        """Log sample predictions.

        Args:
            inputs: Input texts
            predictions: Predicted texts
            targets: Target texts
            num_samples: Number of samples to log
        """
        # Log to console
        self.progress.log_message("Sample predictions:", "info")
        for i in range(min(num_samples, len(inputs))):
            self.progress.log_message(f"  Input: {inputs[i][:50]}...", "debug")
            self.progress.log_message(f"  Pred:  {predictions[i][:50]}...", "debug")
            if targets:
                self.progress.log_message(f"  Target: {targets[i][:50]}...", "debug")

        # Log to WANDB
        if self.use_wandb and self.wandb:
            self.wandb.log_model_predictions(
                inputs, predictions, targets, step=self.global_step, num_samples=num_samples
            )

    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict] = None):
        """Save and log checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            metadata: Additional metadata
        """
        self.progress.set_phase("saving")
        self.progress.log_message(f"Saving checkpoint to {checkpoint_path}", "info")

        # Log to WANDB
        if self.use_wandb and self.wandb:
            self.wandb.log_checkpoint(checkpoint_path, metadata)

        self.progress.log_message("Checkpoint saved", "success")

    def log_data_statistics(self, stats: Dict[str, Any]):
        """Log dataset statistics.

        Args:
            stats: Dataset statistics
        """
        if self.use_wandb and self.wandb:
            self.wandb.log_data_statistics(stats)

    def finish_training(self, save_summary: bool = True):
        """Finish training and generate summary.

        Args:
            save_summary: Whether to save summary to file
        """
        self.progress.set_phase("completed")

        # Calculate total time
        if self.training_start_time is not None:
            total_time = time.time() - self.training_start_time
        else:
            total_time = 0

        # Create summary
        summary = {
            "total_time": total_time,
            "total_time_formatted": self.progress.format_time(total_time),
            "total_epochs": self.epoch,
            "total_steps": self.global_step,
            "best_metric": self.best_metric,
        }

        # Save summary
        if save_summary:
            summary_path = self.output_dir / "training_summary.yaml"
            with open(summary_path, "w") as f:
                yaml.dump(summary, f, default_flow_style=False)

        # Display summary
        self.progress.summary()

        # Finish WANDB run
        if self.use_wandb and self.wandb:
            self.wandb.create_summary_metrics(summary)
            self.wandb.finish()

        self.progress.log_message("Training completed!", "success")

    def __enter__(self):
        """Context manager entry."""
        # Initialize WANDB run if needed
        if self.use_wandb and self.wandb and self.wandb.run is None:
            self.wandb.init_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Handle exceptions
        if exc_type is not None:
            self.progress.set_phase("error")
            self.progress.log_message(f"Training failed: {exc_val}", "error")

        # Finish training
        self.finish_training()


# Convenience function
def create_training_monitor(
    config: Optional[Dict[str, Any]] = None,
    use_wandb: bool = True,
    quiet: bool = False,
    verbose: bool = False,
    **kwargs,
) -> TrainingMonitor:
    """Create a training monitor with sensible defaults.

    Args:
        config: Training configuration
        use_wandb: Enable WANDB tracking
        quiet: Suppress output
        verbose: Detailed output
        **kwargs: Additional arguments for TrainingMonitor

    Returns:
        TrainingMonitor instance
    """
    # Check for WANDB availability
    if use_wandb:
        try:
            import wandb
        except ImportError:
            logger.warning("WANDB not installed, disabling WANDB tracking")
            use_wandb = False

    return TrainingMonitor(config=config, use_wandb=use_wandb, quiet=quiet, verbose=verbose, **kwargs)
