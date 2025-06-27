#!/usr/bin/env python3
"""Unified token counter for arXiv dataset using GPT-NeoX-20B tokenizer.

Combines functionality from count_tokens.py and count_exact_tokens.py.
Supports both sampling-based estimation and exact counting.
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def count_tokens_in_file(file_path: str, tokenizer, sample_size: int = None):
    """Count tokens in a gzipped JSON file.

    Args:
        file_path: Path to the gzipped JSON file
        tokenizer: The tokenizer to use
        sample_size: If provided, only process this many documents

    Returns:
        Dictionary with token statistics
    """
    print(f"{'Sampling' if sample_size else 'Counting all tokens in'} {file_path}")
    if sample_size:
        print(f"Processing first {sample_size} documents for estimation...")
    else:
        print("This may take a few minutes...")

    total_tokens = 0
    doc_count = 0
    token_counts = []
    char_counts = []

    # Try to get total doc count for exact mode
    total_docs = None
    if not sample_size:
        # For arxiv-0098.json.gz we know it's 12966 docs
        if "arxiv-0098" in file_path:
            total_docs = 12966

    with gzip.open(file_path, "rt") as f:
        pbar = tqdm(f, desc="Processing documents", total=total_docs if total_docs else sample_size)

        for i, line in enumerate(pbar):
            if sample_size and i >= sample_size:
                break

            doc = json.loads(line)
            text = doc["text"]

            # Count characters
            char_counts.append(len(text))

            # Tokenize and count
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
            num_tokens = len(tokens)
            token_counts.append(num_tokens)
            total_tokens += num_tokens
            doc_count += 1

            # Update progress bar with stats
            if (i + 1) % 100 == 0:
                avg_tokens = total_tokens / (i + 1)
                pbar.set_postfix({"avg_tokens/doc": f"{avg_tokens:.1f}"})

    return {
        "total_tokens": total_tokens,
        "doc_count": doc_count,
        "token_counts": token_counts,
        "char_counts": char_counts,
        "avg_tokens_per_doc": total_tokens / doc_count if doc_count > 0 else 0,
        "avg_chars_per_doc": sum(char_counts) / len(char_counts) if char_counts else 0,
    }


def print_statistics(stats: dict, sample_size: int = None):
    """Print detailed statistics about the tokenization."""
    print("\n" + "=" * 50)
    print("Token Count Results")
    print("=" * 50)

    if sample_size:
        print(f"Mode: SAMPLING (first {sample_size} documents)")
        # Extrapolate for full dataset if we know the total
        if stats["doc_count"] == sample_size:
            print("\nExtrapolation for full dataset:")
            # Common dataset sizes
            dataset_sizes = {
                "arxiv-0098": 12966,
                "arxiv_sample": 1000,
            }
            # Try to guess dataset size
            for name, size in dataset_sizes.items():
                estimated_total = (stats["total_tokens"] / sample_size) * size
                print(f"  If dataset has {size:,} docs: ~{estimated_total:,.0f} tokens")
    else:
        print("Mode: EXACT (all documents)")

    print(f"\nDocuments processed: {stats['doc_count']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    print(f"Average chars/doc: {stats['avg_chars_per_doc']:.1f}")

    if stats["token_counts"]:
        print("\nToken count statistics:")
        print(f"  Min: {min(stats['token_counts']):,}")
        print(f"  Max: {max(stats['token_counts']):,}")
        print(f"  Median: {np.median(stats['token_counts']):.0f}")
        print(f"  Std: {np.std(stats['token_counts']):.1f}")

        # Character to token ratio
        total_chars = sum(stats["char_counts"])
        char_to_token_ratio = total_chars / stats["total_tokens"]
        print(f"\nCharacter to token ratio: {char_to_token_ratio:.2f} chars/token")


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in dataset using GPT-NeoX-20B tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", type=str, help="Path to the data file (defaults to searching for arxiv data)")
    parser.add_argument("--exact", action="store_true", help="Count all tokens exactly (default: sample first 1000)")
    parser.add_argument(
        "--sample-size", type=int, default=1000, help="Number of documents to sample (ignored if --exact is used)"
    )
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize GPT-NeoX-20B tokenizer (same as OLMo)
    print("Loading GPT-NeoX-20B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Find data file
    if args.file:
        arxiv_file = args.file
    else:
        # Search for arxiv data
        search_paths = [
            "tests/test_data/arxiv-0098.json.gz",
            "../tests/test_data/arxiv-0098.json.gz",
            "../../tests/test_data/arxiv-0098.json.gz",
            "test_data/arxiv-0098.json.gz",
            "data/arxiv-0098.json.gz",
        ]

        arxiv_file = None
        for path in search_paths:
            if os.path.exists(path):
                arxiv_file = path
                break

        if not arxiv_file:
            print("Error: Could not find arxiv data file!")
            print("Searched in:", search_paths)
            print("\nPlease specify file path with --file argument")
            sys.exit(1)

    # Count tokens
    sample_size = None if args.exact else args.sample_size
    stats = count_tokens_in_file(arxiv_file, tokenizer, sample_size)

    # Print results
    print_statistics(stats, sample_size)

    # Save results if requested
    if args.save_results:
        # Remove large arrays from saved results
        save_stats = {
            "file": arxiv_file,
            "mode": "exact" if args.exact else f"sample_{args.sample_size}",
            "total_tokens": stats["total_tokens"],
            "doc_count": stats["doc_count"],
            "avg_tokens_per_doc": stats["avg_tokens_per_doc"],
            "avg_chars_per_doc": stats["avg_chars_per_doc"],
            "tokenizer": "EleutherAI/gpt-neox-20b",
        }

        with open(args.save_results, "w") as f:
            json.dump(save_stats, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
