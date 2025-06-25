"""Training callbacks for OLMo models."""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional
import wandb

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training."""
        pass

    @abstractmethod
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_step_begin(self, trainer, step: int, **kwargs):
        """Called at the beginning of each training step."""
        pass

    @abstractmethod
    def on_step_end(self, trainer, step: int, **kwargs):
        """Called at the end of each training step."""
        pass

    @abstractmethod
    def on_evaluate(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called after evaluation."""
        pass


class LoggingCallback(TrainingCallback):
    """Callback for logging training metrics."""

    def __init__(
        self,
        log_interval: int = 10,
        log_to_wandb: bool = True,
        log_to_tensorboard: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.log_interval = log_interval
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        self.log_dir = log_dir or "./logs"

        self.step_start_time = None
        self.epoch_start_time = None
        self.training_start_time = None

        # Metrics accumulator
        self.accumulated_metrics = {}

    def on_train_begin(self, trainer, **kwargs):
        """Initialize logging at training start."""
        self.training_start_time = time.time()

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize tensorboard writer if needed
        if self.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(self.log_dir)
            except ImportError:
                logger.warning(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )
                self.log_to_tensorboard = False

        logger.info(f"Training started. Logging to {self.log_dir}")

    def on_train_end(self, trainer, **kwargs):
        """Finalize logging at training end."""
        total_time = time.time() - self.training_start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Close tensorboard writer
        if self.log_to_tensorboard and hasattr(self, "tb_writer"):
            self.tb_writer.close()

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Log epoch start."""
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch}")

    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """Log epoch completion."""
        epoch_time = time.time() - self.epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

    def on_step_begin(self, trainer, step: int, **kwargs):
        """Record step start time."""
        self.step_start_time = time.time()

    def on_step_end(self, trainer, step: int, **kwargs):
        """Log training metrics."""
        # Calculate step time
        step_time = time.time() - self.step_start_time

        # Get current metrics
        metrics = kwargs.get("metrics", {})
        metrics["step_time"] = step_time

        # Accumulate metrics
        for key, value in metrics.items():
            if key not in self.accumulated_metrics:
                self.accumulated_metrics[key] = []
            self.accumulated_metrics[key].append(value)

        # Log at specified intervals
        if step % self.log_interval == 0:
            # Calculate averages
            avg_metrics = {
                key: sum(values) / len(values)
                for key, values in self.accumulated_metrics.items()
            }

            # Add step number
            avg_metrics["step"] = step

            # Log to console
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            logger.info(f"Step {step}: {metrics_str}")

            # Log to wandb
            if self.log_to_wandb and wandb.run is not None:
                wandb.log(avg_metrics, step=step)

            # Log to tensorboard
            if self.log_to_tensorboard and hasattr(self, "tb_writer"):
                for key, value in avg_metrics.items():
                    if key != "step":
                        self.tb_writer.add_scalar(f"train/{key}", value, step)

            # Clear accumulated metrics
            self.accumulated_metrics.clear()

    def on_evaluate(self, trainer, metrics: Dict[str, float], **kwargs):
        """Log evaluation metrics."""
        step = kwargs.get("step", 0)

        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Evaluation at step {step}: {metrics_str}")

        # Log to wandb
        if self.log_to_wandb and wandb.run is not None:
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(eval_metrics, step=step)

        # Log to tensorboard
        if self.log_to_tensorboard and hasattr(self, "tb_writer"):
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"eval/{key}", value, step)


class CheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        save_total_limit: Optional[int] = None,
        save_best_only: bool = False,
        metric_for_best: str = "eval_loss",
        mode: str = "min",
    ):
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_total_limit = save_total_limit
        self.save_best_only = save_best_only
        self.metric_for_best = metric_for_best
        self.mode = mode

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints = []

    def on_train_begin(self, trainer, **kwargs):
        """Create checkpoint directory."""
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved to {self.save_dir}")

    def on_train_end(self, trainer, **kwargs):
        """Save final checkpoint."""
        if not self.save_best_only:
            self._save_checkpoint(trainer, "final")

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """No action needed."""
        pass

    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """Optionally save checkpoint at epoch end."""
        pass

    def on_step_begin(self, trainer, step: int, **kwargs):
        """No action needed."""
        pass

    def on_step_end(self, trainer, step: int, **kwargs):
        """Save checkpoint at specified intervals."""
        if not self.save_best_only and step % self.save_interval == 0:
            self._save_checkpoint(trainer, f"step_{step}")

    def on_evaluate(self, trainer, metrics: Dict[str, float], **kwargs):
        """Save checkpoint if best model."""
        if self.save_best_only and self.metric_for_best in metrics:
            current_metric = metrics[self.metric_for_best]

            is_best = (self.mode == "min" and current_metric < self.best_metric) or (
                self.mode == "max" and current_metric > self.best_metric
            )

            if is_best:
                self.best_metric = current_metric
                self._save_checkpoint(trainer, "best")
                logger.info(
                    f"New best model saved with {self.metric_for_best}: {current_metric:.4f}"
                )

    def _save_checkpoint(self, trainer, checkpoint_name: str):
        """Save a checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{checkpoint_name}")

        # Save model and optimizer state
        trainer.save_checkpoint(checkpoint_path)

        # Track saved checkpoints
        if checkpoint_name not in ["best", "final"]:
            self.saved_checkpoints.append(checkpoint_path)

        # Remove old checkpoints if limit is set
        if (
            self.save_total_limit
            and len(self.saved_checkpoints) > self.save_total_limit
        ):
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                import shutil

                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")


class EarlyStoppingCallback(TrainingCallback):
    """Callback for early stopping based on evaluation metrics."""

    def __init__(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.patience_counter = 0
        self.should_stop = False

    def on_train_begin(self, trainer, **kwargs):
        """Reset early stopping state."""
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.patience_counter = 0
        self.should_stop = False

    def on_train_end(self, trainer, **kwargs):
        """No action needed."""
        pass

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Check if training should stop."""
        if self.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            trainer.should_stop = True

    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """No action needed."""
        pass

    def on_step_begin(self, trainer, step: int, **kwargs):
        """Check if training should stop."""
        if self.should_stop:
            logger.info(f"Early stopping triggered at step {step}")
            trainer.should_stop = True

    def on_step_end(self, trainer, step: int, **kwargs):
        """No action needed."""
        pass

    def on_evaluate(self, trainer, metrics: Dict[str, float], **kwargs):
        """Check if metric improved."""
        if self.metric not in metrics:
            return

        current_metric = metrics[self.metric]

        if self.mode == "min":
            improved = current_metric < (self.best_metric - self.min_delta)
        else:
            improved = current_metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"Metric improved to {current_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement. Patience: {self.patience_counter}/{self.patience}"
            )

            if self.patience_counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping patience exhausted")


class GradientAccumulationCallback(TrainingCallback):
    """Callback for handling gradient accumulation."""

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulated_loss = 0.0

    def on_train_begin(self, trainer, **kwargs):
        """Initialize accumulation state."""
        self.accumulated_loss = 0.0

    def on_train_end(self, trainer, **kwargs):
        """No action needed."""
        pass

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """No action needed."""
        pass

    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """No action needed."""
        pass

    def on_step_begin(self, trainer, step: int, **kwargs):
        """No action needed."""
        pass

    def on_step_end(self, trainer, step: int, **kwargs):
        """Handle gradient accumulation."""
        loss = kwargs.get("loss", 0.0)
        self.accumulated_loss += loss

        # Only update weights after accumulation steps
        if (step + 1) % self.accumulation_steps == 0:
            # Log accumulated loss
            avg_loss = self.accumulated_loss / self.accumulation_steps
            kwargs["metrics"] = kwargs.get("metrics", {})
            kwargs["metrics"]["accumulated_loss"] = avg_loss

            # Reset accumulation
            self.accumulated_loss = 0.0

    def on_evaluate(self, trainer, metrics: Dict[str, float], **kwargs):
        """No action needed."""
        pass
