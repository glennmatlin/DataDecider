#!/usr/bin/env python3
"""Unified dataset builder for DataDecider.

Consolidates functionality from:
- build_4m_dataset.py
- build_4m_dataset_fast.py
- quick_build_dataset.py
- prepare_training_data.py

Supports multiple modes, checkpointing, and various dataset sizes.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_decide.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""

    model_size: str = "4m"
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    output_dir: str = "data/processed"

    # Processing options
    mode: str = "standard"  # standard, fast, quick
    batch_size: int = 1000
    num_workers: int = 4

    # Checkpointing
    enable_checkpoint: bool = True
    checkpoint_interval: int = 10000

    # Data options
    validation_split: float = 0.01
    seed: int = 42
    target_tokens: Optional[int] = None
    repeat_data: bool = False

    # Quick mode options (known token counts)
    known_train_sequences: Optional[int] = None
    known_val_sequences: Optional[int] = None


class UnifiedDatasetBuilder:
    """Unified dataset builder combining all functionality."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = self._init_tokenizer()
        self.checkpoint_file = None

        if self.config.enable_checkpoint:
            checkpoint_dir = Path(self.config.output_dir) / ".checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_file = checkpoint_dir / f"{config.model_size}_checkpoint.json"

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        except ValueError as e:
            if "trust_remote_code" in str(e):
                logger.warning("Loading tokenizer with trust_remote_code=True")
                tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, trust_remote_code=True)
            else:
                raise

        # Ensure padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def build(self, input_path: str) -> DatasetDict:
        """Main entry point for building dataset."""

        # Quick mode with known token counts
        if self.config.mode == "quick" and self._can_use_quick_mode():
            return self._build_quick()

        # Load raw data
        raw_data = self._load_raw_data(input_path)

        # Tokenize based on mode
        if self.config.mode == "fast":
            dataset = self._tokenize_fast(raw_data)
        else:
            dataset = self._tokenize_standard(raw_data)

        # Handle data repetition if needed
        if self.config.repeat_data and self.config.target_tokens:
            dataset = self._repeat_to_target_tokens(dataset)

        # Create train/validation split
        dataset_dict = self._create_splits(dataset)

        # Save dataset
        self._save_dataset(dataset_dict)

        return dataset_dict

    def _can_use_quick_mode(self) -> bool:
        """Check if quick mode can be used."""
        return self.config.known_train_sequences is not None and self.config.known_val_sequences is not None

    def _build_quick(self) -> DatasetDict:
        """Quick build with known sequence counts."""
        logger.info("Using quick mode with known sequence counts")

        # Create dummy sequences
        dummy_sequence = [self.tokenizer.eos_token_id] * self.config.max_seq_length

        train_data = {
            "input_ids": [dummy_sequence] * self.config.known_train_sequences,
            "attention_mask": [[1] * self.config.max_seq_length] * self.config.known_train_sequences,
        }

        val_data = {
            "input_ids": [dummy_sequence] * self.config.known_val_sequences,
            "attention_mask": [[1] * self.config.max_seq_length] * self.config.known_val_sequences,
        }

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict(train_data),
                "validation": Dataset.from_dict(val_data),
            }
        )

        return dataset_dict

    def _load_raw_data(self, input_path: str) -> List[Dict]:
        """Load raw data from various formats."""
        path = Path(input_path)

        if path.suffix == ".gz":
            import gzip

            with gzip.open(path, "rt") as f:
                if path.name.endswith(".jsonl.gz"):
                    return [json.loads(line) for line in f]
                else:
                    return json.load(f)
        elif path.suffix == ".jsonl":
            with open(path, "r") as f:
                return [json.loads(line) for line in f]
        elif path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _tokenize_standard(self, raw_data: List[Dict]) -> Dataset:
        """Standard tokenization approach."""
        logger.info("Tokenizing with standard approach")

        # Extract texts
        texts = [doc.get("text", "") for doc in raw_data]

        # Tokenize all texts
        logger.info(f"Tokenizing {len(texts)} documents...")
        all_input_ids = []
        all_attention_masks = []

        for i in tqdm(range(0, len(texts), self.config.batch_size)):
            batch = texts[i : i + self.config.batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            # Process into sequences
            for input_ids in encodings["input_ids"]:
                # Add EOS if needed
                if input_ids[-1] != self.tokenizer.eos_token_id:
                    input_ids.append(self.tokenizer.eos_token_id)

                # Split into sequences
                for j in range(0, len(input_ids), self.config.max_seq_length):
                    seq = input_ids[j : j + self.config.max_seq_length]
                    if len(seq) == self.config.max_seq_length:
                        all_input_ids.append(seq)
                        all_attention_masks.append([1] * len(seq))

        return Dataset.from_dict(
            {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
            }
        )

    def _tokenize_fast(self, raw_data: List[Dict]) -> Dataset:
        """Fast tokenization with checkpointing."""
        logger.info("Tokenizing with fast approach (checkpointing enabled)")

        # Load checkpoint if exists
        start_idx = 0
        all_input_ids = []
        all_attention_masks = []

        if self.checkpoint_file and self.checkpoint_file.exists():
            checkpoint = json.load(open(self.checkpoint_file))
            start_idx = checkpoint["last_index"]
            all_input_ids = checkpoint["input_ids"]
            all_attention_masks = checkpoint["attention_masks"]
            logger.info(f"Resuming from checkpoint at index {start_idx}")

        # Process remaining data
        texts = [doc.get("text", "") for doc in raw_data[start_idx:]]

        for i in tqdm(range(0, len(texts), self.config.batch_size)):
            batch = texts[i : i + self.config.batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            # Process into sequences
            for input_ids in encodings["input_ids"]:
                if input_ids[-1] != self.tokenizer.eos_token_id:
                    input_ids.append(self.tokenizer.eos_token_id)

                for j in range(0, len(input_ids), self.config.max_seq_length):
                    seq = input_ids[j : j + self.config.max_seq_length]
                    if len(seq) == self.config.max_seq_length:
                        all_input_ids.append(seq)
                        all_attention_masks.append([1] * len(seq))

            # Save checkpoint
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(start_idx + i + self.config.batch_size, all_input_ids, all_attention_masks)

        # Clean up checkpoint
        if self.checkpoint_file and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        return Dataset.from_dict(
            {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
            }
        )

    def _save_checkpoint(self, last_index: int, input_ids: List, attention_masks: List):
        """Save checkpoint."""
        checkpoint = {
            "last_index": last_index,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f)
        logger.debug(f"Checkpoint saved at index {last_index}")

    def _repeat_to_target_tokens(self, dataset: Dataset) -> Dataset:
        """Repeat dataset to reach target token count."""
        current_tokens = len(dataset) * self.config.max_seq_length

        if current_tokens >= self.config.target_tokens:
            return dataset

        repeat_factor = self.config.target_tokens // current_tokens + 1
        logger.info(f"Repeating dataset {repeat_factor}x to reach {self.config.target_tokens} tokens")

        # Repeat the data
        repeated_data = {
            "input_ids": dataset["input_ids"] * repeat_factor,
            "attention_mask": dataset["attention_mask"] * repeat_factor,
        }

        # Trim to exact token count
        target_sequences = self.config.target_tokens // self.config.max_seq_length
        repeated_data["input_ids"] = repeated_data["input_ids"][:target_sequences]
        repeated_data["attention_mask"] = repeated_data["attention_mask"][:target_sequences]

        return Dataset.from_dict(repeated_data)

    def _create_splits(self, dataset: Dataset) -> DatasetDict:
        """Create train/validation splits."""
        split = dataset.train_test_split(test_size=self.config.validation_split, seed=self.config.seed)

        return DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )

    def _save_dataset(self, dataset_dict: DatasetDict):
        """Save dataset to disk."""
        output_path = Path(self.config.output_dir) / f"olmo_{self.config.model_size}_dataset"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dataset
        dataset_dict.save_to_disk(str(output_path))

        # Save metadata
        metadata = {
            "model_size": self.config.model_size,
            "tokenizer": self.config.tokenizer_name,
            "max_seq_length": self.config.max_seq_length,
            "train_sequences": len(dataset_dict["train"]),
            "val_sequences": len(dataset_dict["validation"]),
            "total_tokens": (len(dataset_dict["train"]) + len(dataset_dict["validation"])) * self.config.max_seq_length,
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total tokens: {metadata['total_tokens']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset builder for DataDecider", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input_path", help="Path to input data file")
    parser.add_argument("--model-size", default="4m", help="Model size (4m, 10m, 70m, etc)")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b", help="Tokenizer to use")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")

    # Mode selection
    parser.add_argument("--mode", choices=["standard", "fast", "quick"], default="standard", help="Processing mode")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")

    # Checkpointing
    parser.add_argument("--checkpoint", action="store_true", help="Enable checkpointing (fast mode)")
    parser.add_argument("--checkpoint-interval", type=int, default=10000, help="Checkpoint interval")

    # Data options
    parser.add_argument("--validation-split", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-tokens", type=int, help="Target token count (for repetition)")
    parser.add_argument("--repeat-data", action="store_true", help="Repeat data to reach target tokens")

    # Quick mode options
    parser.add_argument("--known-train-sequences", type=int, help="Known train sequences (quick mode)")
    parser.add_argument("--known-val-sequences", type=int, help="Known val sequences (quick mode)")

    args = parser.parse_args()

    # Create config
    config = DatasetConfig(
        model_size=args.model_size,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        enable_checkpoint=args.checkpoint or args.mode == "fast",
        checkpoint_interval=args.checkpoint_interval,
        validation_split=args.validation_split,
        seed=args.seed,
        target_tokens=args.target_tokens,
        repeat_data=args.repeat_data,
        known_train_sequences=args.known_train_sequences,
        known_val_sequences=args.known_val_sequences,
    )

    # Build dataset
    builder = UnifiedDatasetBuilder(config)
    dataset_dict = builder.build(args.input_path)

    # Print summary
    print("\nDataset building complete!")
    print(f"Train sequences: {len(dataset_dict['train']):,}")
    print(f"Validation sequences: {len(dataset_dict['validation']):,}")
    print(f"Total tokens: {(len(dataset_dict['train']) + len(dataset_dict['validation'])) * config.max_seq_length:,}")


if __name__ == "__main__":
    main()
