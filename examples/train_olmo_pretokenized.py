#!/usr/bin/env python3
"""OLMo training script using pre-tokenized datasets.

This script demonstrates the new workflow where tokenization is completely
separated from training, using pre-tokenized datasets from the registry.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import DataDecider components
import numpy as np
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from data_decide.olmo.models import OLMoConfig, OLMoForCausalLM
from data_decide.utils.tokenized_dataset_loader import DatasetRegistry, TokenizedDatasetLoader
from data_decide.utils.training_monitor import create_training_monitor


class PreTokenizedTrainerCallback(TrainerCallback):
    """Custom callback for training monitor integration."""

    def __init__(self, monitor, dataset_info):
        self.monitor = monitor
        self.dataset_info = dataset_info

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.monitor.start_training(
            training_config=args.to_dict(),
            num_epochs=args.num_train_epochs,
            total_steps=state.max_steps,
        )

        # Log dataset information
        self.monitor.log_data_statistics(
            {
                "dataset_name": self.dataset_info.name,
                "total_tokens": self.dataset_info.total_tokens,
                "total_sequences": self.dataset_info.total_sequences,
                "sequence_length": self.dataset_info.sequence_length,
                "tokenizer": self.dataset_info.tokenizer_name,
            }
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        self.monitor.start_epoch(state.epoch, args.num_train_epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        metrics = {"loss": state.log_history[-1].get("loss", 0) if state.log_history else 0}
        self.monitor.end_epoch(metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging metrics."""
        if logs:
            if "loss" in logs:
                self.monitor.update_step(
                    state.global_step,
                    state.max_steps,
                    loss=logs["loss"],
                    metrics=logs,
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            self.monitor.end_evaluation(metrics)

    def on_save(self, args, state, control, **kwargs):
        """Called when saving checkpoint."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.monitor.save_checkpoint(checkpoint_path)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred

    # Calculate perplexity from loss
    loss = np.mean(predictions)
    perplexity = np.exp(loss)

    return {
        "eval_loss": loss,
        "eval_perplexity": perplexity,
    }


def main():
    # Load environment variables first
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train OLMo with pre-tokenized data")

    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="test_small", help="Dataset name from registry or path to dataset"
    )
    parser.add_argument(
        "--registry", type=str, default="configs/dataset_registry.yaml", help="Path to dataset registry"
    )

    # Model configuration
    parser.add_argument("--model-size", type=str, default="4m", help="Model size (4m, 70m, 300m)")
    parser.add_argument("--model-config", type=str, default=None, help="Path to model config YAML")

    # Training configuration
    parser.add_argument("--output-dir", type=str, default="./outputs/olmo-pretokenized", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient-accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # WANDB configuration
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="WANDB project name (defaults to WANDB_PROJECT env var)"
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="WANDB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WANDB tracking")

    # Display options
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (minimal output)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode (detailed output)")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load dataset registry
    registry = DatasetRegistry(args.registry)

    # Load dataset
    if args.dataset in registry.datasets:
        # Load from registry
        dataset_loader = registry.get(args.dataset)
        print(f"Loading dataset '{args.dataset}' from registry")
    else:
        # Load from path
        dataset_loader = TokenizedDatasetLoader(args.dataset)
        print(f"Loading dataset from path: {args.dataset}")

    print(dataset_loader)

    # Load the actual data
    dataset = dataset_loader.load()
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", dataset["train"].select(range(min(100, len(dataset["train"])))))

    # Verify model compatibility
    if args.model_config:
        with open(args.model_config, "r") as f:
            model_config = yaml.safe_load(f)
    else:
        # Default configs
        model_configs = {
            "4m": {
                "hidden_size": 128,
                "num_hidden_layers": 6,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "vocab_size": 50277,  # Must match tokenizer
                "max_position_embeddings": 2048,
            },
            "70m": {
                "hidden_size": 512,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "vocab_size": 50277,
                "max_position_embeddings": 2048,
            },
        }
        model_config = model_configs.get(args.model_size, model_configs["4m"])

    # Verify compatibility
    is_compatible, error_msg = dataset_loader.verify_compatibility(model_config)
    if not is_compatible:
        print(f"WARNING: Dataset compatibility issue: {error_msg}")
        # For GPT-NeoX tokenizer, the actual vocab size is 50254 but we pad to 50277 for efficiency
        if "vocab size" in error_msg and "50254" in error_msg and "50277" in error_msg:
            print("Note: This is expected for GPT-NeoX tokenizer - continuing with padded vocab size")
        else:
            print(f"ERROR: Dataset incompatible with model: {error_msg}")
            sys.exit(1)

    # Create model
    config = OLMoConfig(**model_config)
    model = OLMoForCausalLM(config)

    # Initialize training monitor
    monitor = create_training_monitor(
        config={
            "model_size": args.model_size,
            "dataset": args.dataset,
            "dataset_info": dataset_loader.info.__dict__,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
        },
        use_wandb=not args.no_wandb,
        quiet=args.quiet,
        verbose=args.verbose,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
        or f"olmo-{args.model_size}-{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        output_dir=args.output_dir,
    )

    with monitor:
        # Log model info
        monitor.log_model_info(model)

        # Custom data collator for pre-tokenized data
        def collate_fn(examples):
            # Stack tensors for each field
            batch = {}
            for key in examples[0].keys():
                if key in ["input_ids", "attention_mask", "labels"]:
                    # Pad sequences to same length
                    max_length = max(len(ex[key]) for ex in examples)
                    padded = []
                    for ex in examples:
                        seq = ex[key]
                        if len(seq) < max_length:
                            # Pad with 0 (pad token id)
                            padding = [0] * (max_length - len(seq))
                            if key == "attention_mask":
                                # Attention mask padding is 0
                                seq = seq + padding
                            else:
                                # For input_ids and labels, pad at the end
                                seq = seq + padding
                        padded.append(seq)
                    batch[key] = torch.tensor(padded)
            return batch

        data_collator = collate_fn

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=10,  # Will be limited by max_steps
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps * 2,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            logging_steps=10,
            logging_first_step=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Monitor handles reporting
            max_steps=args.max_steps,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        # Create trainer
        callback = PreTokenizedTrainerCallback(monitor, dataset_loader.info)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

        # Train
        monitor.progress.set_phase("training")
        train_result = trainer.train()

        # Final evaluation
        monitor.progress.set_phase("evaluating")
        eval_results = trainer.evaluate()

        # Save final model
        monitor.progress.set_phase("saving")
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)

        # Log final metrics
        monitor.log_metrics(
            {
                "final_loss": eval_results.get("eval_loss", 0),
                "final_perplexity": eval_results.get("eval_perplexity", 0),
                "total_training_time": train_result.metrics.get("train_runtime", 0),
                "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            }
        )

    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {final_model_path}")
    print(f"Results saved to: {args.output_dir}")

    # Print dataset efficiency metrics
    print("\nDataset Efficiency:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Total tokens used: {dataset_loader.info.total_tokens:,}")
    print(f"  Training samples: {len(train_dataset):,}")
    print("  No tokenization overhead during training!")


if __name__ == "__main__":
    main()
