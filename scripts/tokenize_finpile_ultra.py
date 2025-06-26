#!/usr/bin/env python3
"""Ultra-fast parallel tokenization for FinPile dataset using all available optimizations."""

import argparse
import gc
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import psutil
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from transformers import AutoTokenizer

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class UltraConfig:
    """Configuration for ultra-fast tokenization."""
    input_dir: str
    output_dir: str
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    batch_size: int = 100  # Documents per batch
    sequences_per_chunk: int = 50000  # Larger chunks for better I/O
    num_workers: int = 8  # Parallel workers
    prefetch_files: int = 4  # Files to prefetch
    use_fast: bool = True  # Use fast tokenizer
    
    
class DocumentReader:
    """Efficient document reader with prefetching."""
    
    def __init__(self, file_paths: List[Path], prefetch: int = 4):
        self.file_paths = file_paths
        self.prefetch = prefetch
        
    def read_file_batch(self, file_path: Path, batch_size: int = 100) -> Iterator[List[str]]:
        """Read documents in batches from a gzip file."""
        batch = []
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    text = doc.get("text", "")
                    if text:
                        batch.append(text)
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                            
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        # Yield remaining documents
        if batch:
            yield batch


class BatchTokenizer:
    """Optimized batch tokenizer using fast tokenizers."""
    
    def __init__(self, tokenizer_name: str, max_seq_length: int):
        self.max_seq_length = max_seq_length
        
        # Load fast tokenizer with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,  # Critical for performance
        )
        
        # Configure tokenizer
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Enable parallelism in tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
    def batch_tokenize(self, texts: List[str]) -> List[List[int]]:
        """Tokenize a batch of texts efficiently."""
        # Batch encode with truncation disabled
        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        # Process into sequences
        all_sequences = []
        current_sequence = []
        
        for token_ids in encodings["input_ids"]:
            # Add EOS token
            token_ids.append(self.eos_token_id)
            
            # Add to current sequence
            current_sequence.extend(token_ids)
            
            # Extract complete sequences
            while len(current_sequence) >= self.max_seq_length:
                all_sequences.append(current_sequence[:self.max_seq_length])
                current_sequence = current_sequence[self.max_seq_length:]
        
        # Store remaining tokens for next batch
        self._remainder = current_sequence
        
        return all_sequences
    
    def get_remainder(self) -> Optional[List[int]]:
        """Get remaining tokens that didn't form a complete sequence."""
        if hasattr(self, '_remainder') and self._remainder:
            # Pad if needed
            remainder = self._remainder.copy()
            while len(remainder) < self.max_seq_length:
                remainder.append(self.pad_token_id)
            self._remainder = []
            return remainder[:self.max_seq_length]
        return None


def process_file_worker(args: Tuple[Path, int, UltraConfig]) -> Tuple[int, int, float]:
    """Worker function to process a single file."""
    file_path, file_num, config = args
    
    # Create output path
    output_dir = Path(config.output_dir)
    output_file = output_dir / f"tokenized_{file_num:04d}.arrow"
    
    # Skip if already processed
    if output_file.exists():
        logger.info(f"Skipping already processed: {file_path.name}")
        return file_num, 0, 0.0
    
    start_time = time.time()
    
    # Initialize components
    reader = DocumentReader([file_path])
    tokenizer = BatchTokenizer(config.tokenizer_name, config.max_seq_length)
    
    # Process file
    all_sequences = []
    doc_count = 0
    
    try:
        for doc_batch in reader.read_file_batch(file_path, config.batch_size):
            # Batch tokenize
            sequences = tokenizer.batch_tokenize(doc_batch)
            all_sequences.extend(sequences)
            doc_count += len(doc_batch)
            
            # Save periodically to manage memory
            if len(all_sequences) >= config.sequences_per_chunk:
                # Save to temp file first
                temp_file = output_file.with_suffix('.tmp')
                _save_sequences_fast(all_sequences[:config.sequences_per_chunk], temp_file)
                
                # Clear saved sequences
                all_sequences = all_sequences[config.sequences_per_chunk:]
        
        # Handle remainder
        remainder = tokenizer.get_remainder()
        if remainder:
            all_sequences.append(remainder)
        
        # Save final sequences
        if all_sequences:
            temp_file = output_file.with_suffix('.tmp')
            if temp_file.exists():
                # Append to existing
                existing = Dataset.load_from_disk(str(temp_file))
                all_sequences = existing["input_ids"] + all_sequences
            _save_sequences_fast(all_sequences, temp_file)
        
        # Rename temp to final
        if temp_file.exists():
            temp_file.rename(output_file)
        
        elapsed = time.time() - start_time
        sequence_count = len(Dataset.load_from_disk(str(output_file))["input_ids"])
        
        logger.info(f"Completed {file_path.name}: {sequence_count:,} sequences in {elapsed:.1f}s")
        return file_num, sequence_count, elapsed
        
    except Exception as e:
        logger.error(f"Failed to process {file_path.name}: {e}")
        # Clean up temp file
        temp_file = output_file.with_suffix('.tmp')
        if temp_file.exists():
            temp_file.unlink()
        return file_num, 0, 0.0


def _save_sequences_fast(sequences: List[List[int]], output_file: Path):
    """Save sequences using optimized Arrow format."""
    # Convert to numpy for faster processing
    sequences_array = np.array(sequences, dtype=np.int32)
    
    # Create dataset from numpy array
    dataset = Dataset.from_dict({
        "input_ids": sequences_array.tolist()
    })
    
    # Save with optimized settings
    dataset.save_to_disk(
        str(output_file),
        num_proc=1,  # Single process is faster for I/O
    )


class UltraTokenizer:
    """Main class for ultra-fast parallel tokenization."""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.progress_lock = Lock()
        self.completed_files = self._load_progress()
        
    def _load_progress(self) -> set:
        """Load progress from disk."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get("completed_files", []))
        return set()
    
    def _save_progress(self):
        """Save progress to disk thread-safely."""
        with self.progress_lock:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    "completed_files": list(self.completed_files),
                    "last_update": datetime.now().isoformat()
                }, f, indent=2)
    
    def run(self):
        """Run ultra-fast parallel tokenization."""
        # Get input files
        input_dir = Path(self.config.input_dir)
        all_files = sorted(input_dir.glob("*.json.gz"))
        
        # Filter already processed
        files_to_process = [
            (f, i) for i, f in enumerate(all_files)
            if f.name not in self.completed_files
        ]
        
        logger.info(f"Ultra-fast tokenization starting")
        logger.info(f"Files to process: {len(files_to_process)}/{len(all_files)}")
        logger.info(f"Workers: {self.config.num_workers}")
        logger.info(f"Batch size: {self.config.batch_size}")
        
        if not files_to_process:
            logger.info("All files already processed!")
            return
        
        # Prepare work items
        work_items = [
            (file_path, file_num, self.config)
            for file_path, file_num in files_to_process
        ]
        
        # Process in parallel
        total_sequences = 0
        total_time = 0.0
        completed = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file_worker, item): item[0]
                for item in work_items
            }
            
            # Process results as they complete
            with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        file_num, seq_count, elapsed = future.result()
                        
                        if seq_count > 0:
                            # Update progress
                            self.completed_files.add(file_path.name)
                            self._save_progress()
                            
                            total_sequences += seq_count
                            total_time += elapsed
                            completed += 1
                            
                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({
                                'sequences': f'{total_sequences:,}',
                                'speed': f'{total_sequences/total_time:.0f} seq/s'
                            })
                    
                    except Exception as e:
                        logger.error(f"Error processing {file_path.name}: {e}")
                        pbar.update(1)
        
        # Final statistics
        total_elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("Ultra-fast tokenization complete!")
        logger.info(f"Files processed: {completed}")
        logger.info(f"Total sequences: {total_sequences:,}")
        logger.info(f"Total time: {total_elapsed:.1f}s")
        logger.info(f"Average speed: {total_sequences/total_elapsed:.0f} sequences/second")
        logger.info(f"Speedup: {self.config.num_workers:.1f}x (parallel processing)")
        logger.info("="*60)
        
        # Create final metadata
        self._create_metadata(total_sequences, completed, total_elapsed)
    
    def _create_metadata(self, total_sequences: int, files_processed: int, elapsed: float):
        """Create metadata file."""
        metadata = {
            "tokenizer": self.config.tokenizer_name,
            "max_seq_length": self.config.max_seq_length,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "files_processed": files_processed,
            "total_sequences": total_sequences,
            "processing_time": elapsed,
            "sequences_per_second": total_sequences / elapsed if elapsed > 0 else 0,
            "created_at": datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast parallel tokenization")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=100, help="Documents per batch")
    parser.add_argument("--sequences-per-chunk", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--prefetch-files", type=int, default=4)
    
    args = parser.parse_args()
    
    # Auto-detect optimal workers if not specified
    if args.num_workers == 8:
        cpu_count = mp.cpu_count()
        # Use 80% of CPUs, minimum 4, maximum 16
        args.num_workers = max(4, min(16, int(cpu_count * 0.8)))
    
    config = UltraConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        sequences_per_chunk=args.sequences_per_chunk,
        num_workers=args.num_workers,
        prefetch_files=args.prefetch_files,
    )
    
    logger.info(f"Starting ultra-fast tokenization with {config.num_workers} workers")
    
    tokenizer = UltraTokenizer(config)
    tokenizer.run()


if __name__ == "__main__":
    main()