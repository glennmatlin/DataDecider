#!/usr/bin/env python3
"""Count tokens in arXiv dataset using GPT-NeoX-20B tokenizer."""

import gzip
import json
import os
import sys

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def count_tokens_in_file(file_path: str, tokenizer, sample_size: int = None):
    """Count total tokens in a gzipped JSON file."""

    print(f"Loading data from {file_path}")

    total_tokens = 0
    doc_count = 0
    token_counts = []
    char_counts = []

    with gzip.open(file_path, "rt") as f:
        for i, line in enumerate(tqdm(f, desc="Processing documents")):
            if sample_size and i >= sample_size:
                break

            doc = json.loads(line)
            text = doc["text"]

            # Count characters
            char_counts.append(len(text))

            # Tokenize and count
            tokens = tokenizer.encode(text, truncation=False)
            num_tokens = len(tokens)
            token_counts.append(num_tokens)
            total_tokens += num_tokens
            doc_count += 1

            # Print progress every 1000 docs
            if (i + 1) % 1000 == 0:
                avg_tokens = total_tokens / (i + 1)
                print(f"Processed {i + 1} docs: {total_tokens:,} tokens (avg: {avg_tokens:.1f} tokens/doc)")

    return {
        "total_tokens": total_tokens,
        "doc_count": doc_count,
        "token_counts": token_counts,
        "char_counts": char_counts,
    }


def main():
    # Initialize tokenizer - GPT-NeoX-20B as mentioned in OLMo paper
    print("Loading GPT-NeoX-20B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Find arXiv data
    data_paths = [
        "tests/test_data/arxiv-0098.json.gz",
        "../tests/test_data/arxiv-0098.json.gz",
        "../../tests/test_data/arxiv-0098.json.gz",
    ]

    arxiv_file = None
    for path in data_paths:
        if os.path.exists(path):
            arxiv_file = path
            break

    if not arxiv_file:
        print("Error: Could not find arxiv-0098.json.gz")
        sys.exit(1)

    # Count tokens in full dataset
    print(f"\nCounting tokens in {arxiv_file}")
    results = count_tokens_in_file(arxiv_file, tokenizer)

    # Calculate statistics
    total_tokens = results["total_tokens"]
    doc_count = results["doc_count"]
    token_counts = np.array(results["token_counts"])
    char_counts = np.array(results["char_counts"])

    avg_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    std_tokens = np.std(token_counts)

    print("\n" + "=" * 60)
    print("TOKEN COUNT ANALYSIS")
    print("=" * 60)
    print(f"Total documents: {doc_count:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per document: {avg_tokens:.1f}")
    print(f"Median tokens per document: {median_tokens:.1f}")
    print(f"Std dev tokens: {std_tokens:.1f}")
    print(f"Min tokens: {np.min(token_counts):,}")
    print(f"Max tokens: {np.max(token_counts):,}")

    print(f"\nAverage compression ratio: {np.mean(char_counts) / avg_tokens:.2f} chars/token")

    # Check against 400M requirement
    print("\n" + "=" * 60)
    print("TRAINING DATA REQUIREMENTS")
    print("=" * 60)
    required_tokens = 400_000_000
    print(f"Required tokens for 4M model: {required_tokens:,}")
    print(f"Available tokens: {total_tokens:,}")

    if total_tokens >= required_tokens:
        print(f"✅ SUFFICIENT: {total_tokens / required_tokens:.2f}x the required amount")
    else:
        shortfall = required_tokens - total_tokens
        additional_docs = int(shortfall / avg_tokens)
        print(f"❌ INSUFFICIENT: Need {shortfall:,} more tokens")
        print(f"   (~{additional_docs:,} more documents at current average)")

        # Calculate how many times to repeat data
        repeat_factor = required_tokens / total_tokens
        print(f"\nAlternative: Repeat dataset {repeat_factor:.1f} times")

    # Sample efficient subset
    print("\n" + "=" * 60)
    print("SUBSET RECOMMENDATIONS")
    print("=" * 60)

    # For testing with smaller amounts
    test_sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    for size in test_sizes:
        docs_needed = int(size / avg_tokens)
        print(f"For {size / 1e6:.0f}M tokens: use first {docs_needed:,} documents")


if __name__ == "__main__":
    main()
