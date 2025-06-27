#!/usr/bin/env python3
"""Streaming tokenization script for FinPile dataset - fixes memory issues."""

import argparse
import gc
import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import psutil
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from transformers import AutoTokenizer

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


class StreamingTokenizer:
    """Memory-efficient streaming tokenizer."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        tokenizer_name: str = "EleutherAI/gpt-neox-20b",
        max_seq_length: int = 2048,
        sequences_per_chunk: int = 10000,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.sequences_per_chunk = sequences_per_chunk

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure EOS token
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.completed_files = self._load_progress()

    def _load_progress(self) -> set:
        """Load list of completed files."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                return set(data.get("completed_files", []))
        return set()

    def _save_progress(self):
        """Save progress to disk."""
        with open(self.progress_file, "w") as f:
            json.dump(
                {"completed_files": list(self.completed_files), "last_update": datetime.now().isoformat()}, f, indent=2
            )

    def _tokenize_file(self, file_path: Path) -> Iterator[List[int]]:
        """Tokenize a single file, yielding sequences."""
        current_sequence = []

        with gzip.open(file_path, "rt") as f:
            for line_num, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())
                    text = doc.get("text", "")

                    if text:
                        # Tokenize
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        tokens.append(self.eos_token_id)

                        # Add to current sequence
                        current_sequence.extend(tokens)

                        # Extract complete sequences
                        while len(current_sequence) >= self.max_seq_length:
                            yield current_sequence[: self.max_seq_length]
                            current_sequence = current_sequence[self.max_seq_length :]

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num} in {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")

        # Handle remaining tokens
        if current_sequence:
            # Pad the last sequence
            while len(current_sequence) < self.max_seq_length:
                current_sequence.append(self.pad_token_id)
            yield current_sequence[: self.max_seq_length]

    def process_file(self, file_path: Path, file_num: int):
        """Process a single file and save to disk."""
        if file_path.name in self.completed_files:
            logger.info(f"Skipping already processed: {file_path.name}")
            return

        logger.info(f"Processing {file_path.name} (file {file_num})")
        start_time = time.time()

        # Output file for this input file
        output_file = self.output_dir / f"tokenized_{file_num:04d}.arrow"
        temp_file = output_file.with_suffix(".tmp")

        # Process in chunks
        sequences = []
        total_sequences = 0

        try:
            for sequence in self._tokenize_file(file_path):
                sequences.append(sequence)

                # Save chunk when buffer is full
                if len(sequences) >= self.sequences_per_chunk:
                    self._save_chunk(sequences, temp_file, append=total_sequences > 0)
                    total_sequences += len(sequences)
                    sequences = []

                    # Log progress
                    if total_sequences % 50000 == 0:
                        memory_gb = psutil.Process().memory_info().rss / 1e9
                        logger.info(f"  Saved {total_sequences:,} sequences, Memory: {memory_gb:.1f}GB")

            # Save remaining sequences
            if sequences:
                self._save_chunk(sequences, temp_file, append=total_sequences > 0)
                total_sequences += len(sequences)

            # Rename temp file to final
            temp_file.rename(output_file)

            # Update progress
            self.completed_files.add(file_path.name)
            self._save_progress()

            # Log completion
            elapsed = time.time() - start_time
            logger.info(f"  Completed {file_path.name}: {total_sequences:,} sequences in {elapsed:.1f}s")

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _save_chunk(self, sequences: List[List[int]], output_file: Path, append: bool = False):
        """Save a chunk of sequences to disk."""
        dataset = Dataset.from_dict({"input_ids": sequences})

        if append and output_file.exists():
            # Load existing data
            existing = Dataset.load_from_disk(str(output_file))
            # Concatenate
            dataset = Dataset.from_dict({"input_ids": existing["input_ids"] + sequences})

        # Save to disk
        dataset.save_to_disk(str(output_file))

    def run(self):
        """Process all files in the input directory."""
        # Get all input files
        input_files = sorted(self.input_dir.glob("*.json.gz"))
        total_files = len(input_files)

        logger.info(f"Found {total_files} input files")
        logger.info(f"Already completed: {len(self.completed_files)}")
        logger.info(f"Output directory: {self.output_dir}")

        # Process each file
        for file_num, file_path in enumerate(tqdm(input_files, desc="Processing files")):
            try:
                self.process_file(file_path, file_num)
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Critical error processing {file_path.name}: {e}")
                # Continue with next file

        # Final summary
        logger.info("\nProcessing complete!")
        logger.info(f"Processed files: {len(self.completed_files)}/{total_files}")

        # Create metadata
        self._create_metadata()

    def _create_metadata(self):
        """Create metadata file for the tokenized dataset."""
        metadata = {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_length": self.max_seq_length,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "completed_files": len(self.completed_files),
            "created_at": datetime.now().isoformat(),
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Streaming tokenization for FinPile")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--sequences-per-chunk", type=int, default=10000)

    args = parser.parse_args()

    tokenizer = StreamingTokenizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        sequences_per_chunk=args.sequences_per_chunk,
    )

    tokenizer.run()


if __name__ == "__main__":
    main()
