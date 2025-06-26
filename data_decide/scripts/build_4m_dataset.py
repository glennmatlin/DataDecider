#!/usr/bin/env python3
"""Build a 400M token dataset for OLMo 4M model testing.

This script creates a preprocessed, tokenized dataset that can be
loaded directly for training without re-tokenization.
"""

import gzip
import json
import logging
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class OLMo4MDatasetBuilder:
    """Build a 400M token dataset for 4M model."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.max_seq_length = 2048
        self.target_tokens = 400_000_000  # 400M tokens
        self.output_dir = Path("../data/processed/olmo_4m_400M_tokens")

        # Files to process (in order of preference)
        self.input_files = [
            "../data/raw/arxiv-0098.json.gz",  # ~243M tokens
            "../data/raw/arxiv-0012.json.gz",  # Should give us another ~200M+
            "../data/raw/arxiv-0017.json.gz",  # Backup if needed
        ]

        logger.info(f"Tokenizer: {self.tokenizer.name_or_path}")
        logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Target tokens: {self.target_tokens:,}")

    def build_dataset(self):
        """Build the tokenized dataset."""
        start_time = time.time()

        # Collect tokens until we reach target
        all_token_ids = []
        current_tokens = 0
        current_chunk = []
        total_docs = 0

        logger.info("Starting tokenization...")

        for file_path in self.input_files:
            if current_tokens >= self.target_tokens:
                break

            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            logger.info(f"\nProcessing {file_path.name}...")

            with gzip.open(file_path, "rt") as f:
                pbar = tqdm(desc=f"Tokenizing {file_path.name}", unit="docs")

                for line in f:
                    if current_tokens >= self.target_tokens:
                        break

                    doc = json.loads(line)
                    text = doc["text"]

                    # Tokenize
                    tokens = self.tokenizer.encode(text, truncation=False)

                    # Add EOS token
                    tokens.append(self.tokenizer.eos_token_id)

                    # Add to current chunk
                    current_chunk.extend(tokens)

                    # Extract complete sequences
                    while len(current_chunk) >= self.max_seq_length:
                        sequence = current_chunk[: self.max_seq_length]
                        all_token_ids.append(sequence)
                        current_chunk = current_chunk[self.max_seq_length :]
                        current_tokens += self.max_seq_length

                        # Update progress
                        pbar.set_postfix(
                            {
                                "tokens": f"{current_tokens:,}",
                                "sequences": len(all_token_ids),
                                "progress": f"{current_tokens / self.target_tokens * 100:.1f}%",
                            }
                        )

                    total_docs += 1
                    pbar.update(1)

                pbar.close()

            logger.info(f"Processed {total_docs} documents from {file_path.name}")
            logger.info(f"Current tokens: {current_tokens:,}")

        # Trim to exact target if we went over
        if current_tokens > self.target_tokens:
            sequences_needed = self.target_tokens // self.max_seq_length
            all_token_ids = all_token_ids[:sequences_needed]
            current_tokens = sequences_needed * self.max_seq_length

        elapsed_time = time.time() - start_time

        logger.info("\nTokenization complete!")
        logger.info(f"Total sequences: {len(all_token_ids):,}")
        logger.info(f"Total tokens: {current_tokens:,}")
        logger.info(f"Time elapsed: {elapsed_time:.1f} seconds")

        # Create train/validation split (95/5)
        n_sequences = len(all_token_ids)
        n_val = int(n_sequences * 0.05)
        n_train = n_sequences - n_val

        # Shuffle and split
        indices = np.random.permutation(n_sequences)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create datasets
        train_data = {
            "input_ids": [all_token_ids[i] for i in train_indices],
            "attention_mask": [[1] * self.max_seq_length for _ in range(n_train)],
        }

        val_data = {
            "input_ids": [all_token_ids[i] for i in val_indices],
            "attention_mask": [[1] * self.max_seq_length for _ in range(n_val)],
        }

        # For causal LM, labels are the same as input_ids
        train_data["labels"] = train_data["input_ids"]
        val_data["labels"] = val_data["input_ids"]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)

        dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

        # Save dataset
        self.save_dataset(dataset_dict, total_docs, elapsed_time)

        return dataset_dict

    def save_dataset(self, dataset: DatasetDict, total_docs: int, elapsed_time: float):
        """Save dataset and metadata."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving dataset to {self.output_dir}...")

        # Save dataset
        dataset.save_to_disk(str(self.output_dir))

        # Save metadata
        metadata = {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_length": self.max_seq_length,
            "target_tokens": self.target_tokens,
            "total_documents": total_docs,
            "total_sequences": len(dataset["train"]) + len(dataset["validation"]),
            "train_sequences": len(dataset["train"]),
            "validation_sequences": len(dataset["validation"]),
            "total_train_tokens": len(dataset["train"]) * self.max_seq_length,
            "total_validation_tokens": len(dataset["validation"]) * self.max_seq_length,
            "processing_time_seconds": elapsed_time,
            "files_used": [str(f) for f in self.input_files],
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("DATASET CREATION COMPLETE")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Total sequences: {metadata['total_sequences']:,}")
        print(f"Train sequences: {metadata['train_sequences']:,}")
        print(f"Validation sequences: {metadata['validation_sequences']:,}")
        print(f"Total tokens: {metadata['total_train_tokens'] + metadata['total_validation_tokens']:,}")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print("=" * 70)
        print("\nDataset saved and ready for training!")
        print(f"Load with: Dataset.load_from_disk('{self.output_dir}')")


def main():
    """Build the 4M model dataset."""
    builder = OLMo4MDatasetBuilder()
    dataset = builder.build_dataset()

    # Quick test - load a sample
    print("\nTesting dataset loading...")
    sample = dataset["train"][0]
    print(f"Sample input shape: {len(sample['input_ids'])}")
    print(f"First 10 tokens: {sample['input_ids'][:10]}")
    print("✅ Dataset verified!")


if __name__ == "__main__":
    main()
