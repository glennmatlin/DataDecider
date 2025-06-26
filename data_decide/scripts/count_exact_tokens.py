#!/usr/bin/env python3
"""Count exact tokens in arXiv dataset using GPT-NeoX-20B tokenizer."""

import gzip
import json
import os
import sys

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def count_all_tokens(file_path: str, tokenizer):
    """Count all tokens in the dataset."""

    print(f"Counting tokens in {file_path}")
    print("This may take a few minutes...")

    total_tokens = 0
    doc_count = 0
    token_counts = []

    with gzip.open(file_path, "rt") as f:
        for line in tqdm(f, desc="Processing documents", total=12966):
            doc = json.loads(line)
            text = doc["text"]

            # Tokenize
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            num_tokens = len(tokens)

            token_counts.append(num_tokens)
            total_tokens += num_tokens
            doc_count += 1

    return total_tokens, doc_count, token_counts


def main():
    # Initialize GPT-NeoX-20B tokenizer (same as OLMo)
    print("Loading GPT-NeoX-20B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Find arxiv data
    arxiv_file = None
    paths = [
        "tests/test_data/arxiv-0098.json.gz",
        "../tests/test_data/arxiv-0098.json.gz",
        "../../tests/test_data/arxiv-0098.json.gz",
    ]

    for path in paths:
        if os.path.exists(path):
            arxiv_file = path
            break

    if not arxiv_file:
        print("Error: Could not find arxiv-0098.json.gz")
        sys.exit(1)

    # Count tokens
    total_tokens, doc_count, token_counts = count_all_tokens(arxiv_file, tokenizer)

    # Calculate statistics
    token_counts = np.array(token_counts)

    print("\n" + "=" * 70)
    print("TOKEN COUNT RESULTS")
    print("=" * 70)
    print(f"Total documents: {doc_count:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens/document: {np.mean(token_counts):.1f}")
    print(f"Median tokens/document: {np.median(token_counts):.1f}")
    print(f"Min tokens: {np.min(token_counts):,}")
    print(f"Max tokens: {np.max(token_counts):,}")
    print(f"Std deviation: {np.std(token_counts):.1f}")

    # Check against 4M model requirements
    print("\n" + "=" * 70)
    print("4M MODEL TRAINING REQUIREMENTS")
    print("=" * 70)

    # From the paper: 4M model needs 0.4B tokens
    required_tokens = 400_000_000  # 0.4B
    batch_size = 32
    sequence_length = 2048
    training_steps = 5725

    print(f"Required tokens: {required_tokens:,} (0.4B)")
    print(f"Available tokens: {total_tokens:,}")

    tokens_per_step = batch_size * sequence_length
    total_tokens_in_training = tokens_per_step * training_steps

    print("\nTraining calculation:")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Training steps: {training_steps:,}")
    print(f"Total tokens in training: {total_tokens_in_training:,}")

    if total_tokens >= required_tokens:
        print("\n✅ SUFFICIENT DATA")
        print(f"You have {total_tokens / required_tokens:.2f}x the required tokens")
        print(f"Can train for {total_tokens / tokens_per_step:.0f} steps without repetition")
    else:
        print("\n❌ INSUFFICIENT DATA")
        epochs_needed = required_tokens / total_tokens
        print(f"Need {epochs_needed:.2f} epochs (data repetitions)")
        print(f"Alternative: Train for {total_tokens / tokens_per_step:.0f} steps only")

    # Save results
    results = {
        "file": str(arxiv_file),
        "tokenizer": "EleutherAI/gpt-neox-20b",
        "total_documents": doc_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": float(np.mean(token_counts)),
        "required_tokens_4m": required_tokens,
        "sufficient_data": total_tokens >= required_tokens,
        "epochs_needed": max(1.0, required_tokens / total_tokens),
    }

    results_file = "token_count_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
