#!/usr/bin/env python3
"""Test tokenization on a single file from FinPile dataset."""

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import psutil
from transformers import AutoTokenizer

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


def test_single_file(file_path: str):
    """Test tokenization on a single file to validate setup."""
    logger.info(f"Testing tokenization on: {file_path}")
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Memory tracking
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1e9
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")
    
    # Load and process file
    start_time = time.time()
    total_docs = 0
    total_tokens = 0
    token_lengths = []
    
    logger.info("Processing file...")
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Process only first 1000 documents
                break
                
            try:
                doc = json.loads(line.strip())
                text = doc.get("text", "")
                
                if text:
                    # Tokenize
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_count = len(tokens)
                    
                    total_docs += 1
                    total_tokens += token_count
                    token_lengths.append(token_count)
                    
                    # Log sample
                    if i < 5:
                        logger.info(f"Doc {i}: {token_count} tokens, first 50 chars: {text[:50]}...")
                        
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {i}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1e9
    memory_used = final_memory - initial_memory
    
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_length = max(token_lengths) if token_lengths else 0
    min_length = min(token_lengths) if token_lengths else 0
    
    # Report results
    logger.info("\n" + "="*50)
    logger.info("Test Results:")
    logger.info(f"  Documents processed: {total_docs:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Average doc length: {avg_length:.1f} tokens")
    logger.info(f"  Min/Max length: {min_length:,} / {max_length:,} tokens")
    logger.info(f"  Processing time: {elapsed_time:.1f} seconds")
    logger.info(f"  Memory used: {memory_used:.2f} GB")
    logger.info(f"  Processing rate: {total_docs/elapsed_time:.1f} docs/sec")
    logger.info(f"  Token rate: {total_tokens/elapsed_time:,.0f} tokens/sec")
    
    # Estimate full dataset
    if total_docs > 0:
        docs_per_file = 285000  # Approximate from earlier count
        total_files = 200
        estimated_total_docs = docs_per_file * total_files
        estimated_total_tokens = estimated_total_docs * avg_length
        estimated_time_hours = (estimated_total_docs / (total_docs/elapsed_time)) / 3600
        
        logger.info("\nFull Dataset Estimates:")
        logger.info(f"  Total documents: {estimated_total_docs:,}")
        logger.info(f"  Total tokens: {estimated_total_tokens:,.0f}")
        logger.info(f"  Processing time: {estimated_time_hours:.1f} hours")
        logger.info(f"  Output size: ~{estimated_total_tokens * 2 / 1e9:.1f} GB (assuming 2 bytes/token)")
    
    logger.info("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test tokenization on single file")
    parser.add_argument(
        "--file",
        type=str,
        default="/mnt/z/FinPile/0fp-100dolma/part-00000-b90aa82b-79fb-41f8-ae2c-3b3e17d3d4ae-c000.json.gz",
        help="Path to test file"
    )
    
    args = parser.parse_args()
    test_single_file(args.file)