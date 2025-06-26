#!/usr/bin/env python3
"""Centralized tokenization script for DataDecider.

This script provides a unified interface for tokenizing datasets, supporting:
- Multiple input formats (JSON, JSONL, GZ, text files)
- Configurable output paths and formats
- Progress tracking and resume capability
- Comprehensive metadata generation
"""

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Import DataDecider logging
from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""

    input_path: str
    output_path: str
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    batch_size: int = 1000
    validation_split: float = 0.05
    max_tokens: Optional[int] = None
    append_eos: bool = True
    save_format: str = "arrow"  # arrow or parquet
    num_proc: int = 4
    resume: bool = True

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class DatasetTokenizer:
    """Handles tokenization of various dataset formats."""

    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # Ensure tokenizer has necessary tokens
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id

        # Statistics tracking
        self.stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "total_sequences": 0,
            "avg_doc_length": 0,
            "max_doc_length": 0,
            "min_doc_length": float("inf"),
        }

    def load_documents(self, input_path: Path) -> Iterator[str]:
        """Load documents from various file formats."""
        if input_path.is_dir():
            # Load all files in directory
            files = sorted(input_path.glob("*.json*"))
            for file_path in tqdm(files, desc="Loading files"):
                yield from self._load_single_file(file_path)
        else:
            # Load single file
            yield from self._load_single_file(input_path)

    def _load_single_file(self, file_path: Path) -> Iterator[str]:
        """Load documents from a single file."""
        logger.info(f"Loading {file_path}")

        if file_path.suffix == ".gz":
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"

        with open_fn(file_path, mode) as f:
            if file_path.suffix in [".jsonl", ".gz"]:
                # JSONL format
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        text = doc.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line")
            elif file_path.suffix == ".json":
                # Single JSON file
                data = json.load(f)
                if isinstance(data, list):
                    for doc in data:
                        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
                        if text:
                            yield text
                elif isinstance(data, dict) and "text" in data:
                    yield data["text"]
            else:
                # Plain text file
                yield f.read()

    def tokenize_and_chunk(self, documents: Iterator[str]) -> Dict[str, List]:
        """Tokenize documents and chunk into fixed-length sequences."""
        all_input_ids = []
        current_chunk = []

        for doc in tqdm(documents, desc="Tokenizing"):
            # Update statistics
            self.stats["total_documents"] += 1

            # Tokenize document
            tokens = self.tokenizer.encode(doc, add_special_tokens=False)
            doc_length = len(tokens)

            # Update statistics
            self.stats["total_tokens"] += doc_length
            self.stats["max_doc_length"] = max(self.stats["max_doc_length"], doc_length)
            self.stats["min_doc_length"] = min(self.stats["min_doc_length"], doc_length)

            # Add EOS token if configured
            if self.config.append_eos:
                tokens.append(self.eos_token_id)

            # Add to current chunk
            current_chunk.extend(tokens)

            # Create sequences when chunk is large enough
            while len(current_chunk) >= self.config.max_seq_length:
                sequence = current_chunk[: self.config.max_seq_length]
                all_input_ids.append(sequence)
                current_chunk = current_chunk[self.config.max_seq_length :]
                self.stats["total_sequences"] += 1

                # Check if we've reached max tokens
                if self.config.max_tokens and self.stats["total_tokens"] >= self.config.max_tokens:
                    break

            if self.config.max_tokens and self.stats["total_tokens"] >= self.config.max_tokens:
                break

        # Handle remaining tokens
        if current_chunk and len(current_chunk) == self.config.max_seq_length:
            all_input_ids.append(current_chunk)
            self.stats["total_sequences"] += 1

        # Calculate average document length
        if self.stats["total_documents"] > 0:
            self.stats["avg_doc_length"] = self.stats["total_tokens"] / self.stats["total_documents"]

        # Create attention masks (all 1s for full sequences)
        attention_masks = [[1] * self.config.max_seq_length for _ in all_input_ids]

        # Labels are same as input_ids for causal LM
        labels = all_input_ids.copy()

        return {
            "input_ids": all_input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    def create_dataset(self, tokenized_data: Dict[str, List]) -> DatasetDict:
        """Create HuggingFace dataset with train/validation split."""
        dataset = Dataset.from_dict(tokenized_data)

        # Create train/validation split
        if self.config.validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=self.config.validation_split, seed=42)
            dataset_dict = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
        else:
            dataset_dict = DatasetDict({"train": dataset})

        return dataset_dict

    def save_dataset(self, dataset: DatasetDict, output_path: Path):
        """Save dataset and metadata."""
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dataset
        logger.info(f"Saving dataset to {output_path}")
        if self.config.save_format == "parquet":
            for split, data in dataset.items():
                data.to_parquet(output_path / f"{split}.parquet")
        else:
            dataset.save_to_disk(str(output_path))

        # Prepare metadata
        metadata = {
            "tokenization_config": self.config.to_dict(),
            "tokenizer_info": {
                "name": self.config.tokenizer_name,
                "vocab_size": self.tokenizer.vocab_size,
                "eos_token": self.tokenizer.eos_token,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token": self.tokenizer.pad_token,
            },
            "statistics": self.stats,
            "dataset_info": {
                "train_samples": len(dataset["train"]),
                "validation_samples": len(dataset.get("validation", [])),
                "sequence_length": self.config.max_seq_length,
            },
            "created_at": datetime.now().isoformat(),
            "datadecider_version": "0.1.0",
        }

        # Add checksums for data integrity
        metadata["checksums"] = self._calculate_checksums(output_path)

        # Save metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")

        # Save config as YAML for easy reading
        config_path = output_path / "tokenization_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

    def _calculate_checksums(self, output_path: Path) -> Dict[str, str]:
        """Calculate checksums for all data files."""
        checksums = {}

        for file_path in output_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".arrow", ".parquet"]:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                checksums[str(file_path.relative_to(output_path))] = sha256_hash.hexdigest()

        return checksums

    def run(self):
        """Run the complete tokenization pipeline."""
        input_path = Path(self.config.input_path)
        output_path = Path(self.config.output_path)

        # Check if output already exists
        if output_path.exists() and not self.config.resume:
            logger.warning(f"Output path {output_path} already exists. Use --resume to continue.")
            return

        logger.info("Starting tokenization pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Tokenizer: {self.config.tokenizer_name}")

        # Load documents
        documents = self.load_documents(input_path)

        # Tokenize and chunk
        tokenized_data = self.tokenize_and_chunk(documents)

        logger.info("Tokenization complete:")
        logger.info(f"  Total documents: {self.stats['total_documents']:,}")
        logger.info(f"  Total tokens: {self.stats['total_tokens']:,}")
        logger.info(f"  Total sequences: {self.stats['total_sequences']:,}")
        logger.info(f"  Avg doc length: {self.stats['avg_doc_length']:.1f} tokens")

        # Create dataset
        dataset = self.create_dataset(tokenized_data)

        # Save dataset and metadata
        self.save_dataset(dataset, output_path)

        logger.info("Tokenization pipeline complete!")

        # Print summary
        print("\n" + "=" * 50)
        print("TOKENIZATION SUMMARY")
        print("=" * 50)
        print(f"Output saved to: {output_path}")
        print(f"Total sequences: {self.stats['total_sequences']:,}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        print(f"Train samples: {len(dataset['train']):,}")
        if "validation" in dataset:
            print(f"Validation samples: {len(dataset['validation']):,}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets for DataDecider")
    parser.add_argument("input_path", type=str, help="Input file or directory")
    parser.add_argument("output_path", type=str, help="Output directory for tokenized data")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        help="Tokenizer to use (default: EleutherAI/gpt-neox-20b)",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing (default: 1000)")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Validation split ratio (default: 0.05)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum number of tokens to process")
    parser.add_argument("--no-eos", action="store_true", help="Don't append EOS tokens between documents")
    parser.add_argument(
        "--format", type=str, default="arrow", choices=["arrow", "parquet"], help="Save format (default: arrow)"
    )
    parser.add_argument("--num-proc", type=int, default=4, help="Number of processes for parallel processing")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing output")

    args = parser.parse_args()

    # Create configuration
    config = TokenizationConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        max_tokens=args.max_tokens,
        append_eos=not args.no_eos,
        save_format=args.format,
        num_proc=args.num_proc,
        resume=not args.no_resume,
    )

    # Run tokenization
    tokenizer = DatasetTokenizer(config)
    tokenizer.run()


if __name__ == "__main__":
    main()
