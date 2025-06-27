#!/usr/bin/env python3
"""
Hybrid FinPile tokenizer combining batch processing with memory efficiency.
Uses HuggingFace datasets for speed while maintaining reasonable memory usage.
"""

import argparse
import gc
import gzip
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HybridFinPileTokenizer:
    """
    Hybrid tokenizer that balances speed and memory efficiency.
    - Uses batch tokenization for speed
    - Processes in memory-efficient chunks
    - Supports parallel processing with controlled memory
    """

    def __init__(
        self,
        tokenizer_name: str = "EleutherAI/gpt-neox-20b",
        max_seq_length: int = 2048,
        batch_size: int = 200,  # Smaller than pure batch approach
        chunk_size: int = 10_000,  # Process 10k docs at a time
        sequences_per_save: int = 25_000,  # Save every 25k sequences
    ):
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.sequences_per_save = sequences_per_save

        # Initialize tokenizer
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize tokenizer (needed for each process)."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def read_documents_chunked(self, file_path: Path) -> Iterator[List[Dict]]:
        """Read documents in memory-efficient chunks."""
        chunk = []

        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    if "text" in doc and doc["text"].strip():
                        chunk.append(doc)

                        if len(chunk) >= self.chunk_size:
                            yield chunk
                            chunk = []
                except json.JSONDecodeError:
                    continue

        # Yield remaining documents
        if chunk:
            yield chunk

    def batch_tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Efficiently tokenize a batch of texts.
        This is the key optimization - processing multiple texts at once.
        """
        # Batch tokenization is much faster than individual
        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        all_sequences = []

        for input_ids in encodings["input_ids"]:
            # Add EOS if needed
            if len(input_ids) == 0 or input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)

            # Split into fixed-length sequences
            for i in range(0, len(input_ids), self.max_seq_length):
                sequence = input_ids[i : i + self.max_seq_length]
                if len(sequence) == self.max_seq_length:
                    all_sequences.append(sequence)

        return all_sequences

    def process_file(self, file_path: Path, output_dir: Path, file_idx: int) -> Dict:
        """Process a single file with optimized batch tokenization."""
        start_time = time.time()
        stats = {"file": file_path.name, "documents": 0, "sequences": 0, "tokens": 0, "time": 0, "memory_peak_gb": 0}

        # Initialize tokenizer for this process
        if not hasattr(self, "tokenizer"):
            self._init_tokenizer()

        output_path = output_dir / f"tokenized_{file_idx:04d}.parquet"
        temp_path = output_path.with_suffix(".tmp")

        all_sequences = []
        writer = None

        try:
            # Process file in chunks
            for chunk_idx, document_chunk in enumerate(self.read_documents_chunked(file_path)):
                # Extract texts
                texts = [doc["text"] for doc in document_chunk]
                stats["documents"] += len(texts)

                # Process in batches for optimal speed
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i : i + self.batch_size]
                    sequences = self.batch_tokenize_texts(batch_texts)
                    all_sequences.extend(sequences)

                # Save periodically to control memory
                if len(all_sequences) >= self.sequences_per_save:
                    # Convert to Arrow format
                    table = pa.table({"input_ids": pa.array(all_sequences, type=pa.list_(pa.int32()))})

                    # Write to file
                    if writer is None:
                        writer = pq.ParquetWriter(temp_path, table.schema)
                    writer.write_table(table)

                    stats["sequences"] += len(all_sequences)
                    stats["tokens"] += sum(len(seq) for seq in all_sequences)
                    all_sequences = []

                    # Force garbage collection
                    gc.collect()

                # Log progress
                if chunk_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    docs_per_sec = stats["documents"] / elapsed if elapsed > 0 else 0
                    memory_gb = psutil.Process().memory_info().rss / 1024**3
                    stats["memory_peak_gb"] = max(stats["memory_peak_gb"], memory_gb)
                    logger.info(
                        f"Progress: {stats['documents']:,} docs, {docs_per_sec:.0f} docs/sec, {memory_gb:.1f} GB memory"
                    )

            # Write remaining sequences
            if all_sequences:
                table = pa.table({"input_ids": pa.array(all_sequences, type=pa.list_(pa.int32()))})
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, table.schema)
                writer.write_table(table)
                stats["sequences"] += len(all_sequences)
                stats["tokens"] += sum(len(seq) for seq in all_sequences)

            # Close writer and rename
            if writer:
                writer.close()
            temp_path.rename(output_path)

            # Final stats
            stats["time"] = time.time() - start_time
            stats["docs_per_sec"] = stats["documents"] / stats["time"]

            logger.info(
                f"Completed {file_path.name}: "
                f"{stats['documents']:,} docs in {stats['time']:.1f}s "
                f"({stats['docs_per_sec']:.0f} docs/sec)"
            )

            return stats

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            if writer:
                writer.close()
            if temp_path.exists():
                temp_path.unlink()
            raise
        finally:
            # Clean up
            gc.collect()


def process_file_worker(args):
    """Worker function for parallel processing."""
    file_path, output_dir, file_idx, tokenizer_args = args

    tokenizer = HybridFinPileTokenizer(**tokenizer_args)
    return tokenizer.process_file(file_path, output_dir, file_idx)


def main():
    parser = argparse.ArgumentParser(description="Hybrid FinPile tokenization")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test-files", type=int, help="Process only first N files")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get files to process
    files = sorted(args.input_dir.glob("*.json.gz"))
    if args.test_files:
        files = files[: args.test_files]

    logger.info(f"Found {len(files)} files to process")
    logger.info(f"Using {args.num_workers} workers")

    # Prepare tokenizer args
    tokenizer_args = {
        "tokenizer_name": args.tokenizer,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
    }

    # Process files
    start_time = time.time()
    total_stats = {"documents": 0, "sequences": 0, "tokens": 0, "files": 0}

    if args.num_workers == 1:
        # Single process
        tokenizer = HybridFinPileTokenizer(**tokenizer_args)
        for i, file_path in enumerate(files):
            stats = tokenizer.process_file(file_path, args.output_dir, i)
            for key in ["documents", "sequences", "tokens"]:
                total_stats[key] += stats[key]
            total_stats["files"] += 1
    else:
        # Parallel processing
        work_items = [(file_path, args.output_dir, i, tokenizer_args) for i, file_path in enumerate(files)]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_file_worker, item): item[0] for item in work_items}

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    stats = future.result()
                    for key in ["documents", "sequences", "tokens"]:
                        total_stats[key] += stats[key]
                    total_stats["files"] += 1
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Tokenization Complete!")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_stats['files']}")
    logger.info(f"Documents: {total_stats['documents']:,}")
    logger.info(f"Sequences: {total_stats['sequences']:,}")
    logger.info(f"Tokens: {total_stats['tokens']:,}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Average speed: {total_stats['documents'] / total_time:.0f} docs/sec")


if __name__ == "__main__":
    main()
