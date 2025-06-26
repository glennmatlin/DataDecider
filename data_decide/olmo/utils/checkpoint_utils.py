"""Checkpoint utilities for OLMo training."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    epoch: int,
    step: int,
    best_metric: float,
    checkpoint_dir: str,
    tokenizer=None,
    config: Optional[Dict[str, Any]] = None,
    keep_last_n: int = 3,
) -> str:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        lr_scheduler: Learning rate scheduler state
        epoch: Current epoch
        step: Current training step
        best_metric: Best evaluation metric so far
        checkpoint_dir: Directory to save checkpoint
        tokenizer: Optional tokenizer to save
        config: Optional training configuration
        keep_last_n: Number of recent checkpoints to keep

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_step_{step}_epoch_{epoch}_{timestamp}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model state
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(checkpoint_path)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path)

    # Save training state
    training_state = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
    }

    torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))

    # Save config if provided
    if config is not None:
        with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # Clean up old checkpoints
    cleanup_checkpoints(checkpoint_dir, keep_last_n)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str, model, optimizer=None, lr_scheduler=None, map_location="cpu"
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        lr_scheduler: Optional scheduler to restore state
        map_location: Device to map tensors to

    Returns:
        Dictionary with training state (epoch, step, best_metric)
    """
    # Load model
    model.from_pretrained(checkpoint_path, map_location=map_location)

    # Load training state
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=map_location)

        # Restore optimizer state
        if optimizer is not None and "optimizer_state_dict" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])

        # Restore scheduler state
        if lr_scheduler is not None and "lr_scheduler_state_dict" in training_state:
            if training_state["lr_scheduler_state_dict"] is not None:
                lr_scheduler.load_state_dict(training_state["lr_scheduler_state_dict"])

        return {
            "epoch": training_state.get("epoch", 0),
            "step": training_state.get("step", 0),
            "best_metric": training_state.get("best_metric", float("inf")),
        }

    return {"epoch": 0, "step": 0, "best_metric": float("inf")}


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    if keep_last_n <= 0:
        return

    # Get all checkpoint directories
    checkpoints = []
    for item in Path(checkpoint_dir).iterdir():
        if item.is_dir() and item.name.startswith("checkpoint_"):
            checkpoints.append(item)

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Remove old checkpoints
    for checkpoint in checkpoints[keep_last_n:]:
        import shutil

        shutil.rmtree(checkpoint)
