#!/usr/bin/env python3
"""Quick dataset builder using known token counts."""

import gzip
import json
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    """Build 400M token dataset quickly."""

    print("Quick 400M Token Dataset Builder")
    print("=" * 50)

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # We know arxiv-0098 has ~243M tokens, arxiv-0012 should have similar or more
    # Let's process both to get 400M+ tokens

    files = [
        "../data/raw/arxiv-0098.json.gz",
        "../data/raw/arxiv-0012.json.gz",
    ]

    max_seq_length = 2048
    target_tokens = 400_000_000
    target_sequences = target_tokens // max_seq_length  # 195,312 sequences

    print(f"Target: {target_sequences:,} sequences ({target_tokens:,} tokens)")
    print(f"Processing files: {[Path(f).name for f in files]}")

    all_sequences = []
    total_tokens = 0
    buffer = []

    start_time = time.time()

    for file_path in files:
        if len(all_sequences) >= target_sequences:
            break

        print(f"\nProcessing {Path(file_path).name}...")

        with gzip.open(file_path, "rt") as f:
            pbar = tqdm(desc="Tokenizing", unit="docs")

            for line in f:
                if len(all_sequences) >= target_sequences:
                    break

                doc = json.loads(line)
                tokens = tokenizer.encode(doc["text"], truncation=False, add_special_tokens=True)

                # Add to buffer
                buffer.extend(tokens)

                # Extract complete sequences
                while len(buffer) >= max_seq_length:
                    all_sequences.append(buffer[:max_seq_length])
                    buffer = buffer[max_seq_length:]
                    total_tokens += max_seq_length

                    if len(all_sequences) % 1000 == 0:
                        pbar.set_postfix(
                            {
                                "sequences": f"{len(all_sequences):,}",
                                "progress": f"{len(all_sequences) / target_sequences * 100:.1f}%",
                            }
                        )

                pbar.update(1)

            pbar.close()

        print(f"Current sequences: {len(all_sequences):,}")

    # Trim to exact target
    all_sequences = all_sequences[:target_sequences]

    elapsed = time.time() - start_time
    print(f"\nTokenization complete in {elapsed:.1f} seconds")

    # Create dataset
    print("\nCreating HuggingFace dataset...")

    # Split 95/5
    n_train = int(len(all_sequences) * 0.95)
    n_val = len(all_sequences) - n_train

    # Shuffle
    indices = np.random.permutation(len(all_sequences))

    train_sequences = [all_sequences[i] for i in indices[:n_train]]
    val_sequences = [all_sequences[i] for i in indices[n_train:]]

    # Create datasets
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": train_sequences,
                    "attention_mask": [[1] * max_seq_length] * n_train,
                    "labels": train_sequences,
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "input_ids": val_sequences,
                    "attention_mask": [[1] * max_seq_length] * n_val,
                    "labels": val_sequences,
                }
            ),
        }
    )

    # Save
    output_dir = Path("../data/processed/olmo_4m_400M_tokens")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    # Save metadata
    metadata = {
        "tokenizer": "EleutherAI/gpt-neox-20b",
        "max_seq_length": max_seq_length,
        "total_sequences": len(all_sequences),
        "train_sequences": n_train,
        "validation_sequences": n_val,
        "total_tokens": len(all_sequences) * max_seq_length,
        "processing_time": elapsed,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 50)
    print("DATASET READY!")
    print("=" * 50)
    print(f"Train: {n_train:,} sequences")
    print(f"Val: {n_val:,} sequences")
    print(f"Total: {len(all_sequences) * max_seq_length:,} tokens")
    print(f"Location: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
