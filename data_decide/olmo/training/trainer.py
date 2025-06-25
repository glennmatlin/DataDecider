# src/training/trainer.py
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import json
from typing import Dict, Optional, Any

from ..models.olmo_model import OLMoForCausalLM
from ..models.configuration_olmo import OLMO_CONFIGS
from ..utils.logging_utils import get_logger
from ..utils.checkpoint_utils import save_checkpoint

logger = get_logger(__name__)


class OLMoTrainer:
    """Main trainer class for OLMo models."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[OLMoForCausalLM] = None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
    ):
        self.config = config
        self.training_config = config["training"]

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.training_config["fp16"] else "no",
            gradient_accumulation_steps=self.training_config[
                "gradient_accumulation_steps"
            ],
            log_with=self.training_config["report_to"],
        )

        # Initialize model
        if model is None:
            model_config = OLMO_CONFIGS[self.training_config["model_size"]]
            self.model = OLMoForCausalLM(model_config)
        else:
            self.model = model

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")
        else:
            self.tokenizer = tokenizer

        # Datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Initialize training components
        self._setup_training()

    def _setup_training(self):
        """Setup training components."""
        # Data loaders
        train_sampler = (
            DistributedSampler(
                self.train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )
            if self.accelerator.num_processes > 1
            else None
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config["batch_size"],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.training_config["num_workers"],
            pin_memory=True,
        )

        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.training_config["per_device_eval_batch_size"],
                shuffle=False,
                num_workers=self.training_config["num_workers"],
                pin_memory=True,
            )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        num_training_steps = (
            len(self.train_dataloader) * self.training_config["num_train_epochs"]
        )
        if self.training_config["max_steps"] > 0:
            num_training_steps = min(
                num_training_steps, self.training_config["max_steps"]
            )

        self.lr_scheduler = self._create_scheduler(num_training_steps)

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )

        if self.eval_dataset:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # Initialize tracking
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # Setup logging
        if self.accelerator.is_main_process:
            self._setup_logging()

    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        no_decay = [
            "bias",
            "layer_norm.weight",
            "ln_1.weight",
            "ln_2.weight",
            "norm.weight",
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config["learning_rate"],
            betas=(
                self.training_config["adam_beta1"],
                self.training_config["adam_beta2"],
            ),
            eps=self.training_config["adam_epsilon"],
        )

        return optimizer

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        warmup_steps = self.training_config["warmup_steps"]

        if self.training_config["lr_scheduler_type"] == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.training_config["lr_scheduler_type"] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {self.training_config['lr_scheduler_type']}"
            )

        return scheduler

    def _setup_logging(self):
        """Setup logging and tracking."""
        # Initialize wandb
        if "wandb" in self.training_config["report_to"]:
            wandb.init(
                project="olmo-training",
                name=f"olmo-{self.training_config['model_size']}",
                config=self.config,
            )

        # Create output directory
        self.output_dir = self.config.get("output_dir", "./outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.training_config['model_size']} model")
        logger.info(f"Total training steps: {self.training_config['max_steps']}")

        # Training metrics
        total_loss = 0

        # Training loop
        progress_bar = tqdm(
            total=self.training_config["max_steps"],
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.training_config["num_train_epochs"]):
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.training_config["gradient_accumulation_steps"]
                total_loss += loss.item()

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient accumulation
                if (step + 1) % self.training_config[
                    "gradient_accumulation_steps"
                ] == 0:
                    # Gradient clipping
                    if self.training_config["max_grad_norm"] > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config["max_grad_norm"],
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    progress_bar.update(1)

                    # Logging
                    if self.global_step % self.training_config["logging_steps"] == 0:
                        avg_loss = total_loss / self.training_config["logging_steps"]
                        self._log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": self.lr_scheduler.get_last_lr()[
                                    0
                                ],
                                "train/epoch": epoch,
                                "train/global_step": self.global_step,
                            }
                        )
                        total_loss = 0

                    # Evaluation
                    if (
                        self.global_step % self.training_config["eval_steps"] == 0
                        and self.eval_dataset is not None
                    ):
                        eval_metrics = self.evaluate()
                        self._log_metrics(
                            {f"eval/{k}": v for k, v in eval_metrics.items()}
                        )

                        # Save best model
                        if eval_metrics["loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["loss"]
                            self.save_model(os.path.join(self.output_dir, "best_model"))

                    # Save checkpoint
                    if self.global_step % self.training_config["save_steps"] == 0:
                        self.save_checkpoint()

                    # Check if done
                    if self.global_step >= self.training_config["max_steps"]:
                        logger.info("Reached max steps. Stopping training.")
                        return

        logger.info("Training completed!")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                outputs = self.model(**batch)
                loss = outputs.loss

                # Accumulate metrics
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["attention_mask"].sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(self.eval_dataloader.dataset)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return {"loss": avg_loss, "perplexity": perplexity, "tokens": total_tokens}

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")

        if self.accelerator.is_main_process:
            save_checkpoint(
                model=self.accelerator.unwrap_model(self.model),
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=0,  # Calculate from global_step if needed
                global_step=self.global_step,
                config=self.config,
                checkpoint_dir=checkpoint_dir,
            )
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def save_model(self, output_dir: str):
        """Save the model."""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved model to {output_dir}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to various trackers."""
        if self.accelerator.is_main_process:
            # Log to wandb
            if "wandb" in self.training_config["report_to"]:
                wandb.log(metrics, step=self.global_step)

            # Log to console
            logger.info(f"Step {self.global_step}: {metrics}")
