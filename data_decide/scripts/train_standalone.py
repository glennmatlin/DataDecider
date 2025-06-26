#!/usr/bin/env python3
"""Standalone training script for OLMo 4M model with DataDecide methodology."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add local imports
sys.path.append(str(Path(__file__).parent))

from accelerate import Accelerator
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup

from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS

# Import our OLMo components
from data_decide.olmo.models.olmo_model import OLMoForCausalLM


def setup_logging(output_dir):
    """Simple logging setup."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    return log_file


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_datadecide_metrics(model, eval_dataloader, accelerator):
    """Compute DataDecide proxy metrics during training."""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(len(batch["input_ids"]))))
            if len(losses) > 10:  # Quick evaluation
                break

    losses = torch.cat(losses)
    perplexity = torch.exp(losses.mean())

    # Diversity metric (simplified)
    diversity_score = 0.85  # Placeholder - would compute from data

    # Quality score based on loss convergence
    quality_score = min(0.95, 1.0 / (1.0 + losses.std().item()))

    return {
        "perplexity": perplexity.item(),
        "diversity": diversity_score,
        "quality": quality_score,
        "combined_score": (1.0 / perplexity.item()) * diversity_score * quality_score * 1000,
    }


def main():
    print("OLMo 4M Training with DataDecide Methodology")
    print("=" * 60)

    # Configuration
    config_path = Path("configs/training_configs/olmo_4m_training.yaml")
    training_config = load_config(config_path)["training"]

    # Setup
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(output_dir)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if training_config.get("fp16", False) else "no",
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
    )

    print(f"Device: {accelerator.device}")
    print(f"Mixed precision: {accelerator.mixed_precision}")

    # Load dataset
    print("\nLoading dataset...")
    dataset_path = Path("data/processed/olmo_4m_400M_tokens")
    dataset = DatasetDict.load_from_disk(str(dataset_path))

    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    print(f"Train samples: {len(dataset['train']):,}")
    print(f"Val samples: {len(dataset['validation']):,}")
    print(f"Total tokens: {metadata['total_tokens']:,}")

    # Initialize model
    print("\nInitializing model...")
    config = OLMO_CONFIGS["4M"]
    model = OLMoForCausalLM(config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Initialize tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=training_config["batch_size"],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
    )

    eval_dataloader = DataLoader(
        dataset["validation"],
        batch_size=training_config["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(training_config["adam_beta1"], training_config["adam_beta2"]),
        eps=training_config["adam_epsilon"],
        weight_decay=training_config["weight_decay"],
    )

    # Learning rate scheduler
    num_training_steps = training_config["max_steps"]
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=training_config["warmup_steps"], num_training_steps=num_training_steps
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Training
    print(f"\nStarting training for {num_training_steps} steps...")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Warmup steps: {training_config['warmup_steps']}")

    global_step = 0
    train_losses = []

    # Training loop
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    for epoch in range(100):  # Max epochs (will break on steps)
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / training_config.get("gradient_accumulation_steps", 1)

            # Backward pass
            accelerator.backward(loss)

            # Gradient accumulation
            if (step + 1) % training_config.get("gradient_accumulation_steps", 1) == 0:
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), training_config["max_grad_norm"])

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                train_losses.append(loss.item() * training_config.get("gradient_accumulation_steps", 1))

                # Logging
                if global_step % training_config["logging_steps"] == 0:
                    avg_loss = np.mean(train_losses[-100:])
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"\nStep {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

                # DataDecide metrics
                if global_step % training_config.get("proxy_metrics_steps", 50) == 0:
                    print("\nComputing DataDecide metrics...")
                    metrics = compute_datadecide_metrics(model, eval_dataloader, accelerator)
                    print(
                        f"DataDecide Metrics - Perplexity: {metrics['perplexity']:.2f}, "
                        f"Diversity: {metrics['diversity']:.3f}, Quality: {metrics['quality']:.3f}, "
                        f"Combined: {metrics['combined_score']:.2f}"
                    )

                # Evaluation
                if global_step % training_config["eval_steps"] == 0:
                    model.eval()
                    eval_losses = []
                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                            eval_losses.append(outputs.loss)
                        if len(eval_losses) > 50:
                            break

                    eval_loss = torch.stack(eval_losses).mean()
                    eval_perplexity = torch.exp(eval_loss)
                    print(f"\nEvaluation - Loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}")
                    model.train()

                # Checkpointing
                if global_step % training_config["save_steps"] == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)

                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)

                    if accelerator.is_main_process:
                        unwrapped_model.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        print(f"\nSaved checkpoint to {checkpoint_dir}")

                progress_bar.update(1)

                # Check if done
                if global_step >= num_training_steps:
                    break

        if global_step >= num_training_steps:
            break

    # Save final model
    print("\nSaving final model...")
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # Save training summary
        summary = {
            "model": "OLMo-4M",
            "total_parameters": param_count,
            "training_steps": global_step,
            "final_loss": np.mean(train_losses[-100:]),
            "dataset_tokens": metadata["total_tokens"],
            "training_config": training_config,
            "datadecide_methodology": "Small-scale proxy experiments for data quality prediction",
        }

        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nTraining completed! Model saved to {final_model_dir}")
    print(f"Total steps: {global_step}")
    print(f"Final loss: {np.mean(train_losses[-100:]):.4f}")


if __name__ == "__main__":
    main()
