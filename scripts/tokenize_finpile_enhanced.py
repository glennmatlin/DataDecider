#!/usr/bin/env python3
"""Enhanced tokenization script with checkpointing and monitoring for FinPile dataset."""

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import psutil
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EnhancedTokenizationConfig:
    """Enhanced configuration with checkpointing support."""

    input_path: str
    output_path: str
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    batch_size: int = 1000
    validation_split: float = 0.05
    max_tokens: Optional[int] = None
    append_eos: bool = True
    save_format: str = "arrow"
    num_proc: int = 16
    resume: bool = True
    checkpoint_interval: int = 10  # Save progress every N files
    memory_limit_gb: int = 50  # Max memory usage in GB
    verify_samples: int = 1000  # Number of samples to verify
    error_tolerance: float = 0.01  # Max error rate before stopping
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProcessingState:
    """Track processing state for resume capability."""
    
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    total_documents: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_checkpoint: str = field(default_factory=lambda: datetime.now().isoformat())
    error_count: int = 0
    
    def save(self, path: Path):
        """Save state to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ProcessingState':
        """Load state from JSON file."""
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()


class EnhancedDatasetTokenizer:
    """Enhanced tokenizer with checkpointing and monitoring."""

    def __init__(self, config: EnhancedTokenizationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Ensure tokenizer has necessary tokens
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # State management
        self.output_path = Path(config.output_path)
        self.state_path = self.output_path / "processing_state.json"
        self.state = ProcessingState.load(self.state_path) if config.resume else ProcessingState()
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1e9
        
        # File tracking
        self.processed_files = set(self.state.processed_files)
        self.failed_files = set(self.state.failed_files)

    def check_memory(self):
        """Check memory usage and raise if exceeded."""
        current_memory = self.process.memory_info().rss / 1e9
        memory_used = current_memory - self.initial_memory
        
        if memory_used > self.config.memory_limit_gb:
            raise MemoryError(f"Memory usage ({memory_used:.1f}GB) exceeded limit ({self.config.memory_limit_gb}GB)")
        
        return memory_used

    def get_files_to_process(self, input_path: Path) -> List[Path]:
        """Get list of files to process, excluding already processed ones."""
        all_files = sorted(input_path.glob("*.json.gz"))
        
        # Filter out already processed files
        files_to_process = []
        for file_path in all_files:
            file_name = file_path.name
            if file_name not in self.processed_files and file_name not in self.failed_files:
                files_to_process.append(file_path)
        
        return files_to_process

    def process_file_batch(self, files: List[Path]) -> Dict[str, List]:
        """Process a batch of files and return tokenized data."""
        all_input_ids = []
        current_chunk = []
        
        for file_path in files:
            try:
                logger.info(f"Processing {file_path.name}")
                file_start_time = time.time()
                docs_in_file = 0
                
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        try:
                            doc = json.loads(line.strip())
                            text = doc.get("text", "")
                            
                            if text:
                                # Tokenize document
                                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                                
                                # Add EOS token if configured
                                if self.config.append_eos:
                                    tokens.append(self.eos_token_id)
                                
                                # Add to current chunk
                                current_chunk.extend(tokens)
                                
                                # Check if we need to split into a new sequence
                                while len(current_chunk) >= self.config.max_seq_length:
                                    # Take exactly max_seq_length tokens
                                    all_input_ids.append(current_chunk[:self.config.max_seq_length])
                                    current_chunk = current_chunk[self.config.max_seq_length:]
                                    self.state.total_sequences += 1
                                
                                # Update statistics
                                self.state.total_documents += 1
                                self.state.total_tokens += len(tokens)
                                docs_in_file += 1
                                
                        except json.JSONDecodeError:
                            self.state.error_count += 1
                            
                        # Check memory periodically
                        if docs_in_file % 10000 == 0:
                            memory_used = self.check_memory()
                            if docs_in_file % 50000 == 0:
                                logger.info(f"  Processed {docs_in_file:,} docs, Memory: {memory_used:.1f}GB")
                
                # Mark file as processed
                self.processed_files.add(file_path.name)
                self.state.processed_files.append(file_path.name)
                
                file_time = time.time() - file_start_time
                logger.info(f"  Completed {file_path.name}: {docs_in_file:,} docs in {file_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                self.failed_files.add(file_path.name)
                self.state.failed_files.append(file_path.name)
                self.state.error_count += 1
        
        # Handle remaining tokens in the last chunk
        if current_chunk:
            # Pad the last chunk if needed
            if len(current_chunk) < self.config.max_seq_length:
                current_chunk.extend([self.tokenizer.pad_token_id] * 
                                   (self.config.max_seq_length - len(current_chunk)))
            all_input_ids.append(current_chunk)
            self.state.total_sequences += 1
        
        return {"input_ids": all_input_ids}

    def save_checkpoint(self, tokenized_data: Dict[str, List], checkpoint_num: int):
        """Save current progress as a checkpoint."""
        checkpoint_dir = self.output_path / f"checkpoint_{checkpoint_num:04d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenized data
        dataset = Dataset.from_dict(tokenized_data)
        dataset.save_to_disk(str(checkpoint_dir))
        
        # Update and save state
        self.state.last_checkpoint = datetime.now().isoformat()
        self.state.save(self.state_path)
        
        # Save checkpoint metadata
        metadata = {
            "checkpoint_num": checkpoint_num,
            "sequences": len(tokenized_data["input_ids"]),
            "total_sequences_so_far": self.state.total_sequences,
            "total_tokens_so_far": self.state.total_tokens,
            "total_documents_so_far": self.state.total_documents,
            "processed_files": len(self.processed_files),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_dir / "checkpoint_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved checkpoint {checkpoint_num} with {len(tokenized_data['input_ids']):,} sequences")

    def merge_checkpoints(self) -> DatasetDict:
        """Merge all checkpoints into final dataset."""
        logger.info("Merging checkpoints into final dataset...")
        
        # Find all checkpoint directories
        checkpoint_dirs = sorted([d for d in self.output_path.glob("checkpoint_*") if d.is_dir()])
        
        if not checkpoint_dirs:
            raise ValueError("No checkpoints found to merge")
        
        # Load and merge all checkpoints
        all_sequences = []
        for checkpoint_dir in tqdm(checkpoint_dirs, desc="Loading checkpoints"):
            dataset = Dataset.load_from_disk(str(checkpoint_dir))
            all_sequences.extend(dataset["input_ids"])
        
        # Create train/validation split
        total_sequences = len(all_sequences)
        validation_size = int(total_sequences * self.config.validation_split)
        
        logger.info(f"Total sequences: {total_sequences:,}")
        logger.info(f"Creating {validation_size:,} validation sequences")
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(all_sequences)
        
        train_sequences = all_sequences[validation_size:]
        val_sequences = all_sequences[:validation_size]
        
        # Create datasets
        train_dataset = Dataset.from_dict({"input_ids": train_sequences})
        val_dataset = Dataset.from_dict({"input_ids": val_sequences})
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

    def verify_dataset(self, dataset: DatasetDict):
        """Verify a sample of the dataset for integrity."""
        logger.info(f"Verifying {self.config.verify_samples} random samples...")
        
        errors = 0
        for split_name, split_data in dataset.items():
            # Sample random indices
            num_samples = min(self.config.verify_samples // 2, len(split_data))
            indices = random.sample(range(len(split_data)), num_samples)
            
            for idx in indices:
                sequence = split_data[idx]["input_ids"]
                
                # Check sequence length
                if len(sequence) != self.config.max_seq_length:
                    logger.warning(f"Sequence {idx} in {split_name} has incorrect length: {len(sequence)}")
                    errors += 1
                
                # Check for valid token IDs
                if any(tid >= self.tokenizer.vocab_size for tid in sequence):
                    logger.warning(f"Sequence {idx} in {split_name} contains invalid token IDs")
                    errors += 1
        
        error_rate = errors / self.config.verify_samples
        logger.info(f"Verification complete. Error rate: {error_rate:.2%}")
        
        if error_rate > self.config.error_tolerance:
            raise ValueError(f"Error rate ({error_rate:.2%}) exceeds tolerance ({self.config.error_tolerance:.2%})")

    def run(self):
        """Run the enhanced tokenization pipeline."""
        input_path = Path(self.config.input_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Get files to process
        files_to_process = self.get_files_to_process(input_path)
        total_files = len(files_to_process) + len(self.processed_files)
        
        logger.info(f"Starting tokenization pipeline")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Already processed: {len(self.processed_files)}")
        logger.info(f"Files to process: {len(files_to_process)}")
        
        if not files_to_process:
            logger.info("All files already processed. Moving to merge phase...")
        else:
            # Process files in batches
            checkpoint_num = len(self.processed_files) // self.config.checkpoint_interval
            
            with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                batch_data = {"input_ids": []}
                
                for i in range(0, len(files_to_process), self.config.checkpoint_interval):
                    batch_files = files_to_process[i:i + self.config.checkpoint_interval]
                    
                    # Process batch
                    batch_tokenized = self.process_file_batch(batch_files)
                    batch_data["input_ids"].extend(batch_tokenized["input_ids"])
                    
                    # Save checkpoint
                    checkpoint_num += 1
                    self.save_checkpoint(batch_data, checkpoint_num)
                    batch_data = {"input_ids": []}  # Reset for next batch
                    
                    # Update progress
                    pbar.update(len(batch_files))
                    
                    # Log progress
                    progress_pct = len(self.processed_files) / total_files * 100
                    elapsed_time = datetime.now() - datetime.fromisoformat(self.state.start_time)
                    
                    if len(self.processed_files) > 0:
                        est_total_time = elapsed_time * total_files / len(self.processed_files)
                        est_remaining = est_total_time - elapsed_time
                        
                        logger.info(f"\nProgress: {progress_pct:.1f}% ({len(self.processed_files)}/{total_files} files)")
                        logger.info(f"Elapsed: {elapsed_time}")
                        logger.info(f"Estimated remaining: {est_remaining}")
                        logger.info(f"Total tokens: {self.state.total_tokens:,}")
                        logger.info(f"Total sequences: {self.state.total_sequences:,}")
                        logger.info(f"Memory usage: {self.check_memory():.1f}GB")
                    
                    # Garbage collection
                    gc.collect()
        
        # Merge all checkpoints
        logger.info("\nMerging checkpoints into final dataset...")
        final_dataset = self.merge_checkpoints()
        
        # Verify dataset
        self.verify_dataset(final_dataset)
        
        # Save final dataset
        logger.info("Saving final dataset...")
        final_path = self.output_path / "final"
        final_dataset.save_to_disk(str(final_path))
        
        # Save final metadata
        self.save_final_metadata(final_dataset)
        
        # Clean up checkpoints
        logger.info("Cleaning up checkpoints...")
        for checkpoint_dir in self.output_path.glob("checkpoint_*"):
            import shutil
            shutil.rmtree(checkpoint_dir)
        
        logger.info("Tokenization complete!")
        self.print_summary(final_dataset)

    def save_final_metadata(self, dataset: DatasetDict):
        """Save comprehensive metadata for the final dataset."""
        metadata = {
            "dataset_name": "finpile_0fp_100dolma",
            "tokenization_config": self.config.to_dict(),
            "tokenizer_info": {
                "name": self.config.tokenizer_name,
                "vocab_size": self.tokenizer.vocab_size,
                "eos_token": self.tokenizer.eos_token,
                "eos_token_id": self.tokenizer.eos_token_id,
            },
            "statistics": {
                "total_documents": self.state.total_documents,
                "total_tokens": self.state.total_tokens,
                "total_sequences": self.state.total_sequences,
                "train_sequences": len(dataset["train"]),
                "validation_sequences": len(dataset["validation"]),
                "avg_tokens_per_doc": self.state.total_tokens / self.state.total_documents if self.state.total_documents > 0 else 0,
                "processed_files": len(self.processed_files),
                "failed_files": len(self.failed_files),
                "error_count": self.state.error_count,
            },
            "processing_info": {
                "start_time": self.state.start_time,
                "end_time": datetime.now().isoformat(),
                "total_duration": str(datetime.now() - datetime.fromisoformat(self.state.start_time)),
            },
            "data_info": {
                "sequence_length": self.config.max_seq_length,
                "validation_split": self.config.validation_split,
                "append_eos": self.config.append_eos,
            },
            "created_at": datetime.now().isoformat(),
            "datadecider_version": "0.1.0",
        }
        
        # Save as JSON
        with open(self.output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save config as YAML
        with open(self.output_path / "tokenization_config.yaml", 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

    def print_summary(self, dataset: DatasetDict):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("TOKENIZATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {self.output_path}")
        print(f"Total documents: {self.state.total_documents:,}")
        print(f"Total tokens: {self.state.total_tokens:,}")
        print(f"Total sequences: {self.state.total_sequences:,}")
        print(f"Train sequences: {len(dataset['train']):,}")
        print(f"Validation sequences: {len(dataset['validation']):,}")
        print(f"Failed files: {len(self.failed_files)}")
        print(f"Error count: {self.state.error_count}")
        print(f"Processing time: {datetime.now() - datetime.fromisoformat(self.state.start_time)}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Enhanced tokenization for FinPile dataset")
    parser.add_argument("--input-path", type=str, required=True, help="Input directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Tokenizer name")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--num-proc", type=int, default=16, help="Number of processes")
    parser.add_argument("--no-eos", action="store_true", help="Don't append EOS tokens")
    parser.add_argument("--save-format", type=str, default="arrow", choices=["arrow", "parquet"], help="Save format")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint every N files")
    parser.add_argument("--memory-limit", type=int, default=50, help="Memory limit in GB")
    parser.add_argument("--verify-samples", type=int, default=1000, help="Number of samples to verify")
    parser.add_argument("--error-tolerance", type=float, default=0.01, help="Maximum error tolerance")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without processing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(f"  Input: {args.input_path}")
        print(f"  Output: {args.output_path}")
        print(f"  Tokenizer: {args.tokenizer}")
        print(f"  Max sequence length: {args.max_seq_length}")
        print(f"  Checkpoint interval: {args.checkpoint_interval} files")
        print(f"  Memory limit: {args.memory_limit}GB")
        return
    
    # Create configuration
    config = EnhancedTokenizationConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        append_eos=not args.no_eos,
        save_format=args.save_format,
        num_proc=args.num_proc,
        resume=not args.no_resume,
        checkpoint_interval=args.checkpoint_interval,
        memory_limit_gb=args.memory_limit,
        verify_samples=args.verify_samples,
        error_tolerance=args.error_tolerance,
    )
    
    # Run tokenization
    tokenizer = EnhancedDatasetTokenizer(config)
    tokenizer.run()


if __name__ == "__main__":
    main()