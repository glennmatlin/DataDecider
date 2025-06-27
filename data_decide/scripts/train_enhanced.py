#!/usr/bin/env python3
"""Enhanced training script for OLMo 4M model with rich progress indicators and W&B."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from dotenv import load_dotenv

# Rich imports for enhanced terminal output
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Add local imports
sys.path.append(str(Path(__file__).parent))

from accelerate import Accelerator
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup

from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS

# Import our OLMo components
from data_decide.olmo.models.olmo_model import OLMoForCausalLM

# Load environment variables
load_dotenv()

# Initialize console
console = Console()


def create_training_dashboard():
    """Create a rich dashboard for training metrics."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=7),
        Layout(name="metrics", size=10),
        Layout(name="status", size=3),
    )
    return layout


def update_metrics_table(step, total_steps, loss, lr, perplexity, tokens_per_sec, gpu_memory):
    """Create a table with current training metrics."""
    table = Table(title="Training Metrics", expand=True)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Step", f"{step:,} / {total_steps:,}")
    table.add_row("Progress", f"{(step / total_steps) * 100:.1f}%")
    table.add_row("Loss", f"{loss:.4f}")
    table.add_row("Perplexity", f"{perplexity:.2f}")
    table.add_row("Learning Rate", f"{lr:.2e}")
    table.add_row("Tokens/sec", f"{tokens_per_sec:,.0f}")

    if gpu_memory > 0:
        table.add_row("GPU Memory", f"{gpu_memory:.1f} GB")

    return table


def compute_datadecide_metrics(model, eval_dataloader, accelerator):
    """Compute DataDecide proxy metrics with progress bar."""
    model.eval()
    losses = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        eval_task = progress.add_task("Computing DataDecide metrics...", total=10)

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= 10:  # Quick evaluation
                    break
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather(loss.repeat(len(batch["input_ids"]))))
                progress.update(eval_task, advance=1)

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
    console.print(
        Panel.fit(
            "[bold blue]OLMo 4M Training with DataDecide Methodology[/bold blue]\n"
            "[dim]Enhanced with Weights & Biases tracking and rich progress indicators[/dim]",
            border_style="blue",
        )
    )

    # Configuration
    config_path = Path("configs/training_configs/olmo_4m_training.yaml")
    training_config = yaml.safe_load(open(config_path))["training"]

    # Setup
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    console.print("\n[bold]Initializing Weights & Biases...[/bold]")

    wandb_config = {
        "model": "OLMo-4M",
        "dataset": "arXiv-400M",
        "training_steps": training_config["max_steps"],
        "batch_size": training_config["batch_size"],
        "learning_rate": training_config["learning_rate"],
        "warmup_steps": training_config["warmup_steps"],
        "methodology": "DataDecide",
    }

    run = wandb.init(
        project=training_config.get("wandb_project", "olmo-4m-datadecide"),
        name=training_config.get("wandb_name", f"olmo-4m-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        tags=training_config.get("wandb_tags", ["olmo", "4m", "datadecide"]),
        notes=training_config.get("wandb_notes", ""),
        config=wandb_config,
    )

    console.print(f"[green]✓[/green] W&B run initialized: [link={run.url}]{run.url}[/link]")

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if training_config.get("fp16", False) else "no",
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
    )

    console.print(f"\n[bold]Device:[/bold] {accelerator.device}")
    console.print(f"[bold]Mixed precision:[/bold] {accelerator.mixed_precision}")

    # Load dataset with progress
    console.print("\n[bold]Loading dataset...[/bold]")
    dataset_path = Path("data/processed/olmo_4m_400M_tokens")

    with console.status("[bold green]Loading dataset..."):
        dataset = DatasetDict.load_from_disk(str(dataset_path))

        with open(dataset_path / "metadata.json", "r") as f:
            metadata = json.load(f)

    console.print(f"[green]✓[/green] Train samples: {len(dataset['train']):,}")
    console.print(f"[green]✓[/green] Val samples: {len(dataset['validation']):,}")
    console.print(f"[green]✓[/green] Total tokens: {metadata['total_tokens']:,}")

    # Log dataset info to W&B
    wandb.config.update(
        {
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["validation"]),
            "total_tokens": metadata["total_tokens"],
            "sequence_length": metadata["max_seq_length"],
        }
    )

    # Initialize model
    console.print("\n[bold]Initializing model...[/bold]")
    config = OLMO_CONFIGS["4M"]
    model = OLMoForCausalLM(config)

    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✓[/green] Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Watch model with W&B
    wandb.watch(model, log_freq=100)

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

    # Training setup display
    training_info = Table(title="Training Configuration", show_header=False)
    training_info.add_column("Setting", style="cyan")
    training_info.add_column("Value", style="yellow")

    training_info.add_row("Training steps", f"{num_training_steps:,}")
    training_info.add_row("Learning rate", f"{training_config['learning_rate']}")
    training_info.add_row("Batch size", f"{training_config['batch_size']}")
    training_info.add_row("Warmup steps", f"{training_config['warmup_steps']}")
    training_info.add_row("Weight decay", f"{training_config['weight_decay']}")
    training_info.add_row("Gradient clipping", f"{training_config['max_grad_norm']}")

    console.print("\n", training_info)

    # Initialize metrics
    global_step = 0
    train_losses = []
    start_time = datetime.now()

    # Create main progress bar
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    # Training loop
    model.train()

    with progress:
        main_task = progress.add_task("Training", total=num_training_steps)

        epoch_progress = progress.add_task("Epoch", total=len(train_dataloader))

        for epoch in range(100):  # Max epochs (will break on steps)
            progress.reset(epoch_progress)

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

                    # Calculate metrics
                    avg_loss = np.mean(train_losses[-100:])
                    current_lr = optimizer.param_groups[0]["lr"]
                    perplexity = np.exp(avg_loss)

                    # Calculate tokens per second
                    elapsed = (datetime.now() - start_time).total_seconds()
                    tokens_processed = global_step * training_config["batch_size"] * metadata["max_seq_length"]
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                    # GPU memory (if available)
                    gpu_memory = 0
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.max_memory_allocated() / 1e9

                    # Update progress
                    progress.update(main_task, advance=1)
                    progress.update(epoch_progress, advance=1)

                    # Log to W&B
                    if global_step % training_config["logging_steps"] == 0:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/perplexity": perplexity,
                                "train/learning_rate": current_lr,
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/gpu_memory_gb": gpu_memory,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                            },
                            step=global_step,
                        )

                    # DataDecide metrics
                    if global_step % training_config.get("proxy_metrics_steps", 50) == 0:
                        console.print(f"\n[bold cyan]Computing DataDecide metrics at step {global_step}...[/bold cyan]")
                        metrics = compute_datadecide_metrics(model, eval_dataloader, accelerator)

                        wandb.log(
                            {
                                "datadecide/perplexity": metrics["perplexity"],
                                "datadecide/diversity": metrics["diversity"],
                                "datadecide/quality": metrics["quality"],
                                "datadecide/combined_score": metrics["combined_score"],
                            },
                            step=global_step,
                        )

                        console.print(
                            f"[green]DataDecide Metrics:[/green] "
                            f"Perplexity: {metrics['perplexity']:.2f}, "
                            f"Diversity: {metrics['diversity']:.3f}, "
                            f"Quality: {metrics['quality']:.3f}, "
                            f"Combined: {metrics['combined_score']:.2f}"
                        )

                    # Evaluation
                    if global_step % training_config["eval_steps"] == 0:
                        console.print(f"\n[bold cyan]Evaluating at step {global_step}...[/bold cyan]")
                        model.eval()
                        eval_losses = []

                        eval_progress = progress.add_task("Evaluating", total=min(50, len(eval_dataloader)))

                        for i, eval_batch in enumerate(eval_dataloader):
                            if i >= 50:
                                break
                            with torch.no_grad():
                                outputs = model(**eval_batch)
                                eval_losses.append(outputs.loss)
                            progress.update(eval_progress, advance=1)

                        progress.remove_task(eval_progress)

                        eval_loss = torch.stack(eval_losses).mean()
                        eval_perplexity = torch.exp(eval_loss)

                        wandb.log(
                            {
                                "eval/loss": eval_loss.item(),
                                "eval/perplexity": eval_perplexity.item(),
                            },
                            step=global_step,
                        )

                        console.print(
                            f"[green]Evaluation:[/green] Loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}"
                        )
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

                            # Save to W&B
                            wandb.save(str(checkpoint_dir / "*"))

                            console.print(f"[green]✓[/green] Saved checkpoint to {checkpoint_dir}")

                    # Check if done
                    if global_step >= num_training_steps:
                        break

            if global_step >= num_training_steps:
                break

    # Save final model
    console.print("\n[bold]Saving final model...[/bold]")
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # Save to W&B
        wandb.save(str(final_model_dir / "*"))

        # Save training summary
        summary = {
            "model": "OLMo-4M",
            "total_parameters": param_count,
            "training_steps": global_step,
            "final_loss": np.mean(train_losses[-100:]),
            "dataset_tokens": metadata["total_tokens"],
            "training_time": str(datetime.now() - start_time),
            "training_config": training_config,
            "datadecide_methodology": "Small-scale proxy experiments for data quality prediction",
        }

        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Log summary to W&B
        wandb.summary.update(summary)

    # Finish W&B run
    wandb.finish()

    # Final summary
    console.print(
        Panel.fit(
            f"[bold green]Training Completed![/bold green]\n\n"
            f"Model saved to: {final_model_dir}\n"
            f"Total steps: {global_step:,}\n"
            f"Final loss: {np.mean(train_losses[-100:]):.4f}\n"
            f"Training time: {datetime.now() - start_time}\n\n"
            f"[link={run.url}]View full results on W&B[/link]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
