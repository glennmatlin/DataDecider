"""Optimization utilities for OLMo training."""

from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_epsilon: float = 1e-8,
    no_decay_keywords: Optional[List[str]] = None,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with weight decay handling.

    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        adam_beta1: Beta1 for Adam
        adam_beta2: Beta2 for Adam
        adam_epsilon: Epsilon for Adam
        no_decay_keywords: List of parameter name keywords to exclude from weight decay

    Returns:
        Configured AdamW optimizer
    """
    if no_decay_keywords is None:
        no_decay_keywords = [
            "bias",
            "layer_norm.weight",
            "ln_1.weight",
            "ln_2.weight",
            "norm.weight",
        ]

    # Separate parameters for weight decay
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_keywords) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_keywords) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    # Filter out empty parameter groups
    optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if len(group["params"]) > 0]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    power: float = 1.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ('linear', 'cosine', 'constant', 'polynomial')
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (for cosine scheduler)
        power: Power for polynomial scheduler

    Returns:
        Configured learning rate scheduler
    """
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    elif scheduler_type == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
) -> LambdaLR:
    """
    Create a schedule with polynomial decay from the initial lr to lr_end after warmup.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        lr_end: Final learning rate
        power: Polynomial power

    Returns:
        LambdaLR scheduler with polynomial decay
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / optimizer.defaults["lr"]
        else:
            lr_range = optimizer.defaults["lr"] - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / optimizer.defaults["lr"]

    return LambdaLR(optimizer, lr_lambda)


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get parameter groups for optimizer with proper weight decay handling.

    Args:
        model: The model to get parameters from
        weight_decay: Weight decay coefficient
        no_decay_keywords: List of parameter name keywords to exclude from weight decay

    Returns:
        List of parameter groups for optimizer
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "layer_norm", "ln_", "norm"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def calculate_gradient_norm(parameters: List[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    """
    Calculate gradient norm for monitoring.

    Args:
        parameters: Model parameters
        norm_type: Type of norm (default: 2.0 for L2 norm)

    Returns:
        Gradient norm value
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )

    return total_norm.item()
