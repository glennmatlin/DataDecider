#!/usr/bin/env python3
"""Train OLMo model using GPT-NeoX wrapper on GPU with perplexity evaluation."""

import gzip
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

# Check GPU availability
print("=== OLMo 4M GPU Training (GPT-NeoX Wrapper) ===")
print(f"Time: {datetime.now()}")
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Available: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

# Import required libraries
from datasets import Dataset

# Import our OLMo wrapper
from olmo_wrapper import create_olmo_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

# Set random seed
set_seed(42)


class PerplexityEvaluator:
    """Calculate perplexity for language models."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def calculate_perplexity(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> Dict:
        """Calculate perplexity on a list of texts."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length
                ).to(self.device)

                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])

                # Calculate loss only on non-padded tokens
                mask = inputs["attention_mask"].bool()
                total_loss += (outputs.loss * mask.sum()).item()
                total_tokens += mask.sum().item()

        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = np.exp(avg_loss)

        return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}


def load_sample_data(path: str, num_samples: int = 100) -> List[str]:
    """Load sample data from gzipped JSONL file."""
    texts = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            texts.append(data["text"])
    return texts


def main():
    # Configuration
    MODEL_SIZE = "4M"
    DATA_PATH = "/home/gmatlin/Codespace/DataDecider/data/raw/arxiv_sample.json.gz"
    OUTPUT_DIR = "./olmo_4m_wrapper_gpu_output"
    BATCH_SIZE = 32  # Increased for better GPU utilization
    GRADIENT_ACCUMULATION = 2  # Effective batch size = 64
    LEARNING_RATE = 1.4e-2  # From OLMo paper
    MAX_STEPS = 100
    EVAL_STEPS = 20
    MAX_LENGTH = 512
    FP16 = torch.cuda.is_available()

    print("\nConfiguration:")
    print(f"  Model: OLMo-{MODEL_SIZE} (GPT-NeoX Wrapper)")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  FP16 training: {FP16}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Load data
    print("\nLoading data...")
    texts = load_sample_data(DATA_PATH, num_samples=100)
    print(f"✓ Loaded {len(texts)} documents")

    # Split data
    train_texts = texts[:80]
    eval_texts = texts[80:90]
    test_texts = texts[90:]

    # Create datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Add labels
    train_dataset = train_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    eval_dataset = eval_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    test_dataset = test_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)

    # Create model
    print("\nCreating OLMo 4M model (GPT-NeoX wrapper)...")
    model = create_olmo_model(MODEL_SIZE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count:,} parameters ({param_count / 1e6:.1f}M)")
    print("  Architecture: GPT-NeoX with SwiGLU activation")
    print("  Hidden size: 64, Layers: 8, Heads: 8")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        logging_steps=10,
        logging_first_step=True,
        save_steps=EVAL_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=LEARNING_RATE,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        fp16=FP16,
        bf16=False,
        gradient_checkpointing=False,
        push_to_hub=False,
        report_to=[],  # Disable W&B for now
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Monitor GPU memory
    if torch.cuda.is_available():
        print(f"\nGPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Train
    print("\n" + "=" * 50)
    print("Starting OLMo training (GPT-NeoX wrapper)...")
    print("=" * 50)

    try:
        train_result = trainer.train()
        print("\n✓ Training completed successfully!")

        # Save model
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
        print(f"✓ Model saved to {OUTPUT_DIR}/final_model")

    except torch.cuda.OutOfMemoryError:
        print("\n✗ CUDA out of memory! Try reducing batch_size or sequence length")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Evaluate perplexity
    print("\n" + "=" * 50)
    print("Evaluating perplexity...")
    print("=" * 50)

    evaluator = PerplexityEvaluator(model, tokenizer, device)

    # Evaluate on different sets
    print("\nCalculating perplexity on evaluation sets:")

    # Training set (subset)
    train_ppl = evaluator.calculate_perplexity(train_texts[:10], batch_size=4)
    print("\nTraining set (10 samples):")
    print(f"  Perplexity: {train_ppl['perplexity']:.2f}")
    print(f"  Average loss: {train_ppl['avg_loss']:.4f}")

    # Validation set
    eval_ppl = evaluator.calculate_perplexity(eval_texts, batch_size=4)
    print(f"\nValidation set ({len(eval_texts)} samples):")
    print(f"  Perplexity: {eval_ppl['perplexity']:.2f}")
    print(f"  Average loss: {eval_ppl['avg_loss']:.4f}")

    # Test set
    test_ppl = evaluator.calculate_perplexity(test_texts, batch_size=4)
    print(f"\nTest set ({len(test_texts)} samples):")
    print(f"  Perplexity: {test_ppl['perplexity']:.2f}")
    print(f"  Average loss: {test_ppl['avg_loss']:.4f}")

    # Simple benchmark texts
    benchmark_texts = [
        "The financial markets showed significant volatility today.",
        "Machine learning models require large amounts of data.",
        "The Federal Reserve announced a new monetary policy.",
        "Neural networks can approximate complex functions.",
        "Stock prices fluctuated throughout the trading session.",
    ]

    benchmark_ppl = evaluator.calculate_perplexity(benchmark_texts, batch_size=1)
    print("\nBenchmark sentences:")
    print(f"  Perplexity: {benchmark_ppl['perplexity']:.2f}")
    print(f"  Average loss: {benchmark_ppl['avg_loss']:.4f}")

    # Test generation
    print("\n" + "=" * 50)
    print("Testing generation...")
    print("=" * 50)

    model.eval()
    prompts = ["The financial markets", "Machine learning models", "The stock market today"]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=50, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")

    # Final GPU memory report
    if torch.cuda.is_available():
        print(f"\n\nFinal GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 50)
    print("OLMo training and evaluation complete!")
    print("=" * 50)

    # Save results
    results = {
        "model_size": MODEL_SIZE,
        "model_type": "OLMo-GPTNeoX-Wrapper",
        "architecture": "GPTNeoXForCausalLM with SwiGLU",
        "parameters": param_count,
        "training_steps": MAX_STEPS,
        "final_train_loss": train_result.metrics.get("train_loss", None),
        "perplexity": {
            "train": train_ppl["perplexity"],
            "eval": eval_ppl["perplexity"],
            "test": test_ppl["perplexity"],
            "benchmark": benchmark_ppl["perplexity"],
        },
        "device": str(device),
        "fp16": FP16,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
