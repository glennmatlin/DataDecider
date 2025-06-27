#!/usr/bin/env python3
"""Verify the 400M token dataset is ready for training."""

import json
from pathlib import Path

from datasets import DatasetDict


def main():
    """Verify dataset loading and statistics."""

    dataset_path = Path("../data/processed/olmo_4m_400M_tokens")

    print("Verifying OLMo 4M Dataset")
    print("=" * 50)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = DatasetDict.load_from_disk(str(dataset_path))

    # Load metadata
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Train samples: {len(dataset['train']):,}")
    print(f"Validation samples: {len(dataset['validation']):,}")
    print(f"Total sequences: {metadata['total_sequences']:,}")
    print(f"Total tokens: {metadata['total_tokens']:,}")
    print(f"Sequence length: {metadata['max_seq_length']}")
    print(f"Tokenizer: {metadata['tokenizer']}")

    # Check sample
    print("\nSample Check:")
    sample = dataset["train"][0]
    print(f"Input shape: {len(sample['input_ids'])}")
    print(f"Attention mask shape: {len(sample['attention_mask'])}")
    print(f"Labels shape: {len(sample['labels'])}")
    print(f"First 10 tokens: {sample['input_ids'][:10]}")

    # Memory estimate
    total_params = metadata["total_sequences"] * metadata["max_seq_length"]
    memory_gb = (total_params * 2) / (1024**3)  # 2 bytes per int16
    print(f"\nEstimated dataset size in memory: {memory_gb:.2f} GB")

    print("\n✅ Dataset verified and ready for training!")
    print(f"Load with: datasets.load_from_disk('{dataset_path}')")


if __name__ == "__main__":
    main()
