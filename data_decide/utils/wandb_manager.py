"""WANDB manager for comprehensive experiment tracking in DataDecider."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil
import torch
import wandb
import yaml

# Optional imports
try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class WANDBManager:
    """Manages WANDB integration for DataDecider experiments."""

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        dir: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
    ):
        """Initialize WANDB manager.

        Args:
            project: WANDB project name (defaults to WANDB_PROJECT env var)
            entity: WANDB entity (defaults to WANDB_ENTITY env var)
            config: Configuration dictionary
            tags: List of tags for the run
            name: Run name
            notes: Run notes
            mode: WANDB mode (online, offline, disabled)
            dir: Directory to save WANDB files
            group: Group runs together
            job_type: Type of job (train, eval, etc.)
        """
        # Load from environment if not provided
        self.project = project or os.getenv("WANDB_PROJECT", "datadecider")
        self.entity = entity or os.getenv("WANDB_ENTITY")
        self.config = config or {}
        self.tags = tags or []
        self.name = name
        self.notes = notes
        self.mode = mode
        self.dir = dir or "./wandb"
        self.group = group
        self.job_type = job_type

        self.run = None
        self.step = 0
        self.system_metrics_enabled = True

        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        self.gpus = []
        if self.gpu_available and GPUTIL_AVAILABLE:
            try:
                self.gpus = GPUtil.getGPUs()
            except Exception as e:
                logger.debug(f"Could not get GPU info: {e}")

    def init_run(self, reinit: bool = False):
        """Initialize a WANDB run."""
        # Load environment variables if available
        if os.path.exists(".env"):
            from dotenv import load_dotenv

            load_dotenv()
            # Update project/entity from env vars if they weren't explicitly set
            if self.project == "datadecider":  # Default value
                self.project = os.getenv("WANDB_PROJECT", "datadecider")
            if self.entity is None:
                self.entity = os.getenv("WANDB_ENTITY")

        # Add DataDecider specific tags
        self.tags.extend(["datadecider", f"mode_{self.mode}"])

        # Initialize run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            name=self.name,
            notes=self.notes,
            mode=self.mode,
            dir=self.dir,
            group=self.group,
            job_type=self.job_type,
            reinit=reinit,
        )

        # Log system info
        self._log_system_info()

        # Log model architecture if available
        if "model_config" in self.config:
            self._log_model_architecture()

        logger.info(f"WANDB run initialized: {self.run.name} ({self.run.id})")
        logger.info(f"View run at: {self.run.url}")

        return self.run

    def _log_system_info(self):
        """Log system information."""
        system_info = {
            "python_version": os.sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        }

        if torch.cuda.is_available():
            system_info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                }
            )

        wandb.config.update({"system": system_info})

    def _log_model_architecture(self):
        """Log model architecture details."""
        model_config = self.config.get("model_config", {})

        # Create architecture summary
        arch_summary = {
            "model_type": model_config.get("model_type", "unknown"),
            "parameters": model_config.get("num_parameters", 0),
            "layers": model_config.get("num_layers", 0),
            "hidden_size": model_config.get("hidden_size", 0),
            "attention_heads": model_config.get("num_attention_heads", 0),
        }

        wandb.config.update({"architecture": arch_summary})

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """Log metrics to WANDB.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number (uses internal counter if None)
            commit: Whether to commit the log immediately
        """
        if self.run is None:
            return

        if step is None:
            step = self.step
            self.step += 1

        # Add system metrics if enabled
        if self.system_metrics_enabled:
            metrics.update(self._get_system_metrics())

        # Log to WANDB
        wandb.log(metrics, step=step, commit=commit)

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {
            "system/cpu_percent": psutil.cpu_percent(),
            "system/memory_percent": psutil.virtual_memory().percent,
        }

        if self.gpu_available:
            try:
                # PyTorch GPU metrics
                for i in range(torch.cuda.device_count()):
                    metrics[f"system/gpu_{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                    metrics[f"system/gpu_{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024**3)

                # GPUtil metrics if available
                if self.gpus:
                    for i, gpu in enumerate(self.gpus):
                        metrics[f"system/gpu_{i}_utilization"] = gpu.load * 100
                        metrics[f"system/gpu_{i}_temperature"] = gpu.temperature
            except Exception as e:
                logger.debug(f"Error getting GPU metrics: {e}")

        return metrics

    def log_gradients(self, model: torch.nn.Module, step: Optional[int] = None):
        """Log model gradients.

        Args:
            model: PyTorch model
            step: Step number
        """
        if self.run is None:
            return

        # Collect gradient statistics
        grad_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[f"gradients/{name}_mean"] = grad.mean().item()
                grad_stats[f"gradients/{name}_std"] = grad.std().item()
                grad_stats[f"gradients/{name}_max"] = grad.max().item()
                grad_stats[f"gradients/{name}_min"] = grad.min().item()

        self.log_metrics(grad_stats, step=step)

    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """Log learning rate from optimizer.

        Args:
            optimizer: PyTorch optimizer
            step: Step number
        """
        if self.run is None:
            return

        lrs = {}
        for i, param_group in enumerate(optimizer.param_groups):
            lrs[f"train/learning_rate_group_{i}"] = param_group["lr"]

        self.log_metrics(lrs, step=step)

    def log_model_predictions(
        self,
        inputs: List[str],
        predictions: List[str],
        targets: Optional[List[str]] = None,
        step: Optional[int] = None,
        num_samples: int = 10,
    ):
        """Log model predictions as a table.

        Args:
            inputs: Input texts
            predictions: Predicted texts
            targets: Target texts (optional)
            step: Step number
            num_samples: Number of samples to log
        """
        if self.run is None:
            return

        # Create table
        columns = ["Input", "Prediction"]
        data = []

        if targets:
            columns.append("Target")

        # Sample predictions
        indices = np.random.choice(len(inputs), min(num_samples, len(inputs)), replace=False)

        for idx in indices:
            row = [inputs[idx], predictions[idx]]
            if targets:
                row.append(targets[idx])
            data.append(row)

        # Log table
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"predictions": table}, step=step)

    def log_data_statistics(
        self,
        dataset_stats: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """Log dataset statistics.

        Args:
            dataset_stats: Dictionary of dataset statistics
            step: Step number
        """
        if self.run is None:
            return

        # Create visualizations for data statistics
        stats_to_log = {}

        for key, value in dataset_stats.items():
            if isinstance(value, (int, float)):
                stats_to_log[f"data/{key}"] = value
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                # Create histogram
                stats_to_log[f"data/{key}_hist"] = wandb.Histogram(value)

        self.log_metrics(stats_to_log, step=step)

    def log_perplexity_curve(
        self,
        perplexities: List[float],
        steps: List[int],
        name: str = "perplexity",
    ):
        """Log perplexity curve.

        Args:
            perplexities: List of perplexity values
            steps: List of step numbers
            name: Name for the curve
        """
        if self.run is None:
            return

        # Create line plot
        data = [[step, ppl] for step, ppl in zip(steps, perplexities)]
        table = wandb.Table(data=data, columns=["step", name])

        wandb.log({f"{name}_curve": wandb.plot.line(table, "step", name, title=f"{name.capitalize()} over Training")})

    def log_checkpoint(
        self,
        checkpoint_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Additional metadata
        """
        if self.run is None:
            return

        # Create artifact
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata=metadata or {},
        )

        # Add checkpoint file
        artifact.add_file(checkpoint_path)

        # Log artifact
        self.run.log_artifact(artifact)
        logger.info(f"Logged checkpoint: {checkpoint_path}")

    def log_config_file(self, config_path: str):
        """Log configuration file.

        Args:
            config_path: Path to config file
        """
        if self.run is None:
            return

        # Save config file
        wandb.save(config_path)

        # Also parse and log contents
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config = json.load(f)
            else:
                return

        wandb.config.update({"config_file": config})

    def create_summary_metrics(self, metrics: Dict[str, float]):
        """Create summary metrics for the run.

        Args:
            metrics: Dictionary of summary metrics
        """
        if self.run is None:
            return

        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def finish(self, exit_code: int = 0):
        """Finish the WANDB run.

        Args:
            exit_code: Exit code (0 for success)
        """
        if self.run is None:
            return

        # Log final metrics
        self.create_summary_metrics(
            {
                "exit_code": exit_code,
                "total_steps": self.step,
                "finished_at": datetime.now().isoformat(),
            }
        )

        # Finish run
        wandb.finish(exit_code=exit_code)
        logger.info("WANDB run finished")

    def __enter__(self):
        """Context manager entry."""
        if self.run is None:
            self.init_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        exit_code = 0 if exc_type is None else 1
        self.finish(exit_code=exit_code)


# Convenience functions
def init_wandb_run(project: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> WANDBManager:
    """Initialize a WANDB run with DataDecider defaults.

    Args:
        project: Project name (defaults to WANDB_PROJECT env var)
        config: Configuration dictionary
        **kwargs: Additional arguments for WANDBManager

    Returns:
        Initialized WANDBManager
    """
    manager = WANDBManager(project=project, config=config, **kwargs)
    manager.init_run()
    return manager
