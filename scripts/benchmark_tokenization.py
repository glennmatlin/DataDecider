#!/usr/bin/env python3
"""Benchmark different tokenization approaches to find the fastest."""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_test_documents(file_path: Path, num_docs: int = 1000) -> List[str]:
    """Load test documents from a file."""
    documents = []
    
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= num_docs:
                break
            try:
                doc = json.loads(line.strip())
                text = doc.get("text", "")
                if text:
                    documents.append(text)
            except:
                continue
    
    return documents


def benchmark_single_tokenization(tokenizer, documents: List[str]) -> Tuple[float, int]:
    """Benchmark single document tokenization."""
    start_time = time.time()
    total_tokens = 0
    
    for doc in documents:
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        total_tokens += len(tokens)
    
    elapsed = time.time() - start_time
    return elapsed, total_tokens


def benchmark_batch_tokenization(tokenizer, documents: List[str], batch_size: int = 32) -> Tuple[float, int]:
    """Benchmark batch tokenization."""
    start_time = time.time()
    total_tokens = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Batch encode
        encodings = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        for tokens in encodings["input_ids"]:
            total_tokens += len(tokens)
    
    elapsed = time.time() - start_time
    return elapsed, total_tokens


def benchmark_fast_tokenizer(documents: List[str]) -> dict:
    """Benchmark fast tokenizer performance."""
    # Load fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        use_fast=True,
    )
    
    results = {}
    
    # Single document
    elapsed, tokens = benchmark_single_tokenization(tokenizer, documents)
    results["single_fast"] = {
        "time": elapsed,
        "tokens": tokens,
        "docs_per_sec": len(documents) / elapsed,
        "tokens_per_sec": tokens / elapsed,
    }
    
    # Batch sizes
    for batch_size in [8, 16, 32, 64, 100, 200]:
        elapsed, tokens = benchmark_batch_tokenization(tokenizer, documents, batch_size)
        results[f"batch_fast_{batch_size}"] = {
            "time": elapsed,
            "tokens": tokens,
            "docs_per_sec": len(documents) / elapsed,
            "tokens_per_sec": tokens / elapsed,
        }
    
    return results


def print_results(results: dict):
    """Print benchmark results in a nice format."""
    print("\n" + "="*80)
    print("TOKENIZATION BENCHMARK RESULTS")
    print("="*80)
    
    # Sort by tokens per second
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["tokens_per_sec"],
        reverse=True
    )
    
    print(f"\n{'Method':<20} {'Time (s)':<10} {'Docs/sec':<12} {'Tokens/sec':<15} {'Speedup':<10}")
    print("-"*77)
    
    baseline_speed = sorted_results[-1][1]["tokens_per_sec"]
    
    for method, stats in sorted_results:
        speedup = stats["tokens_per_sec"] / baseline_speed
        print(f"{method:<20} {stats['time']:<10.2f} {stats['docs_per_sec']:<12.0f} "
              f"{stats['tokens_per_sec']:<15.0f} {speedup:<10.1f}x")
    
    # Best configuration
    best_method, best_stats = sorted_results[0]
    print(f"\nBest configuration: {best_method}")
    print(f"Speed: {best_stats['tokens_per_sec']:,.0f} tokens/second")
    print(f"Speedup over baseline: {best_stats['tokens_per_sec']/baseline_speed:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokenization performance")
    parser.add_argument(
        "--input-file",
        type=str,
        default="/mnt/z/FinPile/0fp-100dolma/part-00000-b90aa82b-79fb-41f8-ae2c-3b3e17d3d4ae-c000.json.gz",
        help="Input file to benchmark"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=1000,
        help="Number of documents to benchmark"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Loading {args.num_docs} documents for benchmarking...")
    documents = load_test_documents(Path(args.input_file), args.num_docs)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Run benchmarks
    logger.info("Running benchmarks...")
    results = benchmark_fast_tokenizer(documents)
    
    # Print results
    print_results(results)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. Use batch tokenization with optimal batch size (usually 32-100)")
    print("2. Enable fast tokenizers (use_fast=True)")
    print("3. Disable unnecessary outputs (attention_mask, token_type_ids)")
    print("4. Use parallel processing across multiple files")
    print("5. Optimize I/O with larger chunk sizes")


if __name__ == "__main__":
    main()