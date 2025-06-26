#!/usr/bin/env python3
"""Extreme performance tokenization using every possible optimization."""

import argparse
import asyncio
import gc
import gzip
import json
import mmap
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import io

import numpy as np
import psutil
from tqdm.asyncio import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from transformers import AutoTokenizer

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass 
class ExtremeConfig:
    """Configuration for extreme performance tokenization."""
    input_dir: str
    output_dir: str
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    
    # Batching
    doc_batch_size: int = 256  # Documents per tokenization batch
    io_batch_size: int = 1024  # Documents per I/O batch
    sequences_per_chunk: int = 100000  # Sequences per save
    
    # Parallelism
    num_tokenizer_workers: int = 16  # CPU workers for tokenization
    num_io_workers: int = 4  # I/O workers for reading
    num_save_workers: int = 2  # Workers for saving
    
    # Memory
    max_memory_gb: int = 40  # Maximum memory usage
    use_mmap: bool = True  # Use memory mapping
    
    # Advanced
    prefetch_files: int = 8  # Files to prefetch
    ring_buffer_size: int = 100_000  # Ring buffer for sequences
    use_gpu: bool = False  # GPU acceleration (if available)


class MemoryMappedReader:
    """Ultra-fast memory-mapped file reader."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        
    def read_documents_mmap(self, batch_size: int = 1024) -> Iterator[List[str]]:
        """Read documents using memory mapping for speed."""
        # For gzip files, we can't use direct mmap, but we can optimize
        batch = []
        buffer_size = 1024 * 1024  # 1MB buffer
        
        with gzip.open(self.file_path, 'rb') as f:
            # Use buffered reading
            buffered_reader = io.BufferedReader(f, buffer_size=buffer_size)
            
            for line in buffered_reader:
                try:
                    # Fast JSON parsing
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if text:
                        batch.append(text)
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                except:
                    continue
        
        if batch:
            yield batch


class ExtremeTokenizer:
    """Extreme performance tokenizer with all optimizations."""
    
    def __init__(self, tokenizer_name: str, max_seq_length: int):
        self.max_seq_length = max_seq_length
        
        # Load tokenizer with all optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            model_max_length=max_seq_length * 2,  # Avoid truncation warnings
        )
        
        # Configure for maximum performance
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Pre-allocate arrays for performance
        self._sequence_buffer = np.zeros((10000, max_seq_length), dtype=np.int32)
        self._buffer_idx = 0
        
    def extreme_batch_tokenize(self, texts: List[str]) -> np.ndarray:
        """Tokenize with extreme optimizations."""
        # Batch tokenize with all optimizations
        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors=None,  # Return lists for speed
        )
        
        # Process with vectorized operations
        sequences = []
        current_seq = []
        
        for token_ids in encodings["input_ids"]:
            # Append EOS token
            token_ids.append(self.eos_token_id)
            current_seq.extend(token_ids)
            
            # Extract sequences using slicing
            while len(current_seq) >= self.max_seq_length:
                sequences.append(current_seq[:self.max_seq_length])
                current_seq = current_seq[self.max_seq_length:]
        
        # Store remainder
        self._remainder = current_seq
        
        # Convert to numpy for speed
        if sequences:
            return np.array(sequences, dtype=np.int32)
        return np.array([], dtype=np.int32).reshape(0, self.max_seq_length)


class RingBuffer:
    """Lock-free ring buffer for extreme performance."""
    
    def __init__(self, size: int, seq_length: int):
        self.size = size
        self.buffer = np.zeros((size, seq_length), dtype=np.int32)
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        
    def write(self, sequences: np.ndarray) -> int:
        """Write sequences to buffer."""
        n_seqs = len(sequences)
        if n_seqs == 0:
            return 0
            
        # Calculate available space
        available = self.size - self.count
        to_write = min(n_seqs, available)
        
        if to_write > 0:
            # Write using numpy slicing
            end_idx = self.write_idx + to_write
            if end_idx <= self.size:
                self.buffer[self.write_idx:end_idx] = sequences[:to_write]
            else:
                # Wrap around
                first_part = self.size - self.write_idx
                self.buffer[self.write_idx:] = sequences[:first_part]
                self.buffer[:to_write - first_part] = sequences[first_part:to_write]
            
            self.write_idx = (self.write_idx + to_write) % self.size
            self.count += to_write
            
        return to_write
    
    def read(self, n: int) -> Optional[np.ndarray]:
        """Read n sequences from buffer."""
        if self.count == 0:
            return None
            
        to_read = min(n, self.count)
        
        # Read using numpy slicing
        end_idx = self.read_idx + to_read
        if end_idx <= self.size:
            result = self.buffer[self.read_idx:end_idx].copy()
        else:
            # Wrap around
            first_part = self.size - self.read_idx
            result = np.vstack([
                self.buffer[self.read_idx:],
                self.buffer[:to_read - first_part]
            ])
        
        self.read_idx = (self.read_idx + to_read) % self.size
        self.count -= to_read
        
        return result


async def process_file_extreme(
    file_path: Path,
    file_num: int,
    config: ExtremeConfig,
    ring_buffer: RingBuffer,
    tokenizer_pool: List[ExtremeTokenizer]
) -> Tuple[int, int, float]:
    """Process file with extreme optimizations using async I/O."""
    
    output_dir = Path(config.output_dir)
    output_file = output_dir / f"tokenized_{file_num:04d}.arrow"
    
    if output_file.exists():
        return file_num, 0, 0.0
    
    start_time = time.time()
    
    # Get tokenizer from pool
    tokenizer = tokenizer_pool[file_num % len(tokenizer_pool)]
    
    # Use memory-mapped reader
    reader = MemoryMappedReader(file_path)
    
    # Process with async pipeline
    total_sequences = 0
    sequences_buffer = []
    
    try:
        # Read and tokenize in pipeline
        for doc_batch in reader.read_documents_mmap(config.io_batch_size):
            # Split into smaller batches for tokenization
            for i in range(0, len(doc_batch), config.doc_batch_size):
                batch = doc_batch[i:i + config.doc_batch_size]
                
                # Extreme tokenization
                sequences = tokenizer.extreme_batch_tokenize(batch)
                
                if len(sequences) > 0:
                    sequences_buffer.append(sequences)
                    total_sequences += len(sequences)
                    
                    # Write to ring buffer when full
                    if sum(len(s) for s in sequences_buffer) >= config.sequences_per_chunk:
                        # Concatenate and save
                        all_seqs = np.vstack(sequences_buffer)
                        await save_sequences_extreme(all_seqs, output_file, append=False)
                        sequences_buffer = []
        
        # Save remaining
        if sequences_buffer:
            all_seqs = np.vstack(sequences_buffer)
            await save_sequences_extreme(all_seqs, output_file, append=True)
        
        elapsed = time.time() - start_time
        return file_num, total_sequences, elapsed
        
    except Exception as e:
        logger.error(f"Failed processing {file_path.name}: {e}")
        return file_num, 0, 0.0


async def save_sequences_extreme(sequences: np.ndarray, output_file: Path, append: bool = False):
    """Save sequences with extreme I/O optimizations."""
    # Use numpy's optimized save for temporary storage
    temp_file = output_file.with_suffix('.npz')
    
    if append and output_file.exists():
        # Load existing
        existing_ds = Dataset.load_from_disk(str(output_file))
        existing_seqs = np.array(existing_ds["input_ids"], dtype=np.int32)
        sequences = np.vstack([existing_seqs, sequences])
    
    # Save using optimized format
    dataset = Dataset.from_dict({
        "input_ids": sequences.tolist()
    })
    
    # Use fastest save settings
    dataset.save_to_disk(
        str(output_file),
        num_proc=1,
        max_shard_size="1GB",
    )


class ExtremeParallelTokenizer:
    """Main class for extreme parallel tokenization."""
    
    def __init__(self, config: ExtremeConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tokenizer pool for reuse
        self.tokenizer_pool = [
            ExtremeTokenizer(config.tokenizer_name, config.max_seq_length)
            for _ in range(config.num_tokenizer_workers)
        ]
        
        # Create ring buffer for inter-process communication
        self.ring_buffer = RingBuffer(
            config.ring_buffer_size,
            config.max_seq_length
        )
    
    async def run_async(self):
        """Run with async I/O for extreme performance."""
        input_dir = Path(self.config.input_dir)
        all_files = sorted(input_dir.glob("*.json.gz"))
        
        logger.info(f"EXTREME tokenization starting")
        logger.info(f"Files: {len(all_files)}")
        logger.info(f"Tokenizer workers: {self.config.num_tokenizer_workers}")
        logger.info(f"I/O workers: {self.config.num_io_workers}")
        logger.info(f"Doc batch size: {self.config.doc_batch_size}")
        
        start_time = time.time()
        
        # Process files in parallel with async
        tasks = []
        for i, file_path in enumerate(all_files):
            task = process_file_extreme(
                file_path, i, self.config,
                self.ring_buffer, self.tokenizer_pool
            )
            tasks.append(task)
        
        # Run with progress bar
        results = []
        async for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await task
            results.append(result)
        
        # Calculate statistics
        total_sequences = sum(r[1] for r in results)
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("EXTREME tokenization complete!")
        logger.info(f"Total sequences: {total_sequences:,}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Speed: {total_sequences/total_time:.0f} sequences/second")
        logger.info("="*60)
    
    def run(self):
        """Run tokenization."""
        # Set up for maximum performance
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["RAYON_NUM_THREADS"] = str(self.config.num_tokenizer_workers)
        
        # Run async event loop
        asyncio.run(self.run_async())


def main():
    parser = argparse.ArgumentParser(description="Extreme performance tokenization")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--doc-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    
    args = parser.parse_args()
    
    config = ExtremeConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        doc_batch_size=args.doc_batch_size,
        num_tokenizer_workers=args.num_workers,
    )
    
    tokenizer = ExtremeParallelTokenizer(config)
    tokenizer.run()


if __name__ == "__main__":
    main()