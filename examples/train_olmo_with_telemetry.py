#!/usr/bin/env python3
"""OLMo 4M training script with comprehensive telemetry and monitoring."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import DataDecider components
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from data_decide.olmo.models import OLMoConfig, OLMoForCausalLM
from data_decide.utils.training_monitor import create_training_monitor


def load_telemetry_config(config_path: str = "configs/telemetry_config.yaml") -> Dict:
    """Load telemetry configuration."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            "progress": {"enabled": True, "quiet": False, "verbose": False},
            "wandb": {"enabled": True, "mode": "online", "project": "datadecider"},
            "monitoring": {"track_peak_memory": True},
        }


def load_data(data_path: str, tokenizer, max_samples: int = 1000) -> Dataset:
    """Load and prepare dataset."""
    import gzip

    documents = []
    data_file = Path(data_path) / "arxiv_0.jsonl.gz"

    if data_file.exists():
        with gzip.open(data_file, "rt") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                doc = json.loads(line)
                documents.append(doc["text"][:1000])  # Truncate for demo
    else:
        # Generate synthetic data
        documents = [f"Sample text {i} for training." * 10 for i in range(max_samples)]

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    dataset = Dataset.from_dict({"text": documents})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)

    return tokenized_dataset


def create_model_config(model_size: str = "4m") -> OLMoConfig:
    """Create model configuration."""
    if model_size == "4m":
        return OLMoConfig(
            hidden_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            intermediate_size=512,
            vocab_size=50277,
            max_position_embeddings=512,
            head_dim=32,
            rope_theta=10000.0,
            layer_norm_eps=1e-5,
            hidden_act="gelu",
            use_cache=False,
            tie_word_embeddings=True,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred

    # Calculate perplexity
    loss = np.mean(predictions)
    perplexity = np.exp(loss)

    return {
        "eval_loss": loss,
        "eval_perplexity": perplexity,
    }


class TelemetryTrainerCallback(TrainerCallback):
    """Custom callback to integrate with TrainingMonitor."""

    def __init__(self, monitor):
        self.monitor = monitor

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.monitor.start_training(
            training_config=args.to_dict(),
            num_epochs=args.num_train_epochs,
            total_steps=state.max_steps,
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        self.monitor.start_epoch(state.epoch, args.num_train_epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        metrics = {
            "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
        }
        self.monitor.end_epoch(metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging metrics."""
        if logs:
            # Update step progress
            if "loss" in logs:
                self.monitor.update_step(
                    state.global_step,
                    state.max_steps,
                    loss=logs["loss"],
                    metrics=logs,
                )

            # Log learning rate
            if "learning_rate" in logs and hasattr(self, "optimizer"):
                self.monitor.log_learning_rate(self.optimizer)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            self.monitor.end_evaluation(metrics)

    def on_save(self, args, state, control, **kwargs):
        """Called when saving checkpoint."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.monitor.save_checkpoint(checkpoint_path)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Training monitor will be closed by context manager
        pass


def main():
    parser = argparse.ArgumentParser(description="Train OLMo with telemetry")
    parser.add_argument("--model-size", type=str, default="4m", help="Model size")
    parser.add_argument("--data-path", type=str, default="./data", help="Data path")
    parser.add_argument("--output-dir", type=str, default="./outputs/olmo-telemetry", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--eval-steps", type=int, default=20, help="Evaluation frequency")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="olmo-datadecider", help="WANDB project")
    parser.add_argument("--wandb-name", type=str, default=None, help="WANDB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WANDB")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--telemetry-config", type=str, default="configs/telemetry_config.yaml", help="Telemetry config"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Load telemetry config
    telemetry_config = load_telemetry_config(args.telemetry_config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize training monitor
    monitor = create_training_monitor(
        config={
            "model_size": args.model_size,
            "data_path": args.data_path,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
        },
        use_wandb=not args.no_wandb and telemetry_config["wandb"]["enabled"],
        quiet=args.quiet or not telemetry_config["progress"]["enabled"],
        verbose=args.verbose or telemetry_config["progress"]["verbose"],
        wandb_project=args.wandb_project or telemetry_config["wandb"]["project"],
        wandb_name=args.wandb_name or f"olmo-{args.model_size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        wandb_mode=telemetry_config["wandb"]["mode"],
        output_dir=args.output_dir,
    )

    with monitor:
        # Load tokenizer
        monitor.progress.log_message("Loading tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token

        # Load data
        monitor.progress.log_message("Loading dataset...", "info")
        dataset = load_data(args.data_path, tokenizer)

        # Split dataset
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        # Log data statistics
        monitor.log_data_statistics(
            {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "vocab_size": tokenizer.vocab_size,
            }
        )

        # Create model
        monitor.progress.log_message("Creating model...", "info")
        config = create_model_config(args.model_size)
        model = OLMoForCausalLM(config)

        # Log model info
        monitor.log_model_info(model)

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps * 2,
            warmup_steps=10,
            learning_rate=args.learning_rate,
            logging_steps=10,
            logging_first_step=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # We handle reporting through our monitor
            max_steps=args.max_steps,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        # Create trainer with telemetry callback
        callback = TelemetryTrainerCallback(monitor)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

        # Store optimizer reference in callback
        callback.optimizer = trainer.optimizer

        # Train
        monitor.progress.set_phase("training")
        trainer.train()

        # Final evaluation
        monitor.progress.set_phase("evaluating")
        eval_results = trainer.evaluate()

        # Generate sample predictions
        monitor.progress.log_message("Generating sample predictions...", "info")
        model.eval()

        sample_inputs = [
            "The financial markets",
            "Machine learning models",
            "In this paper we propose",
        ]

        predictions = []
        for text in sample_inputs:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                )
            predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

        monitor.log_predictions(sample_inputs, predictions)

        # Save final model
        monitor.progress.set_phase("saving")
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Log final metrics
        monitor.log_metrics(
            {
                "final_loss": eval_results.get("eval_loss", 0),
                "final_perplexity": eval_results.get("eval_perplexity", 0),
            }
        )

    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {final_model_path}")
    print(f"Logs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
