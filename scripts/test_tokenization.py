#!/usr/bin/env python3
"""
Test tokenization setup before full production run.
Validates configuration and estimates performance.
"""

import gzip
import json
import time
from pathlib import Path

from transformers import AutoTokenizer


def test_tokenizer_setup():
    """Test tokenizer configuration"""
    print("Testing tokenizer setup...")

    # Test different tokenizer options
    tokenizers = [
        "EleutherAI/gpt-neox-20b",  # Original
        "allenai/OLMo-1B",  # Recommended for OLMo
    ]

    for tokenizer_name in tokenizers:
        try:
            print(f"\nTesting {tokenizer_name}:")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

            # Test tokenization
            test_text = "This is a test of the tokenization system."
            tokens = tokenizer.encode(test_text)

            print("  ✓ Loaded successfully")
            print(f"  - Vocab size: {len(tokenizer)}")
            print(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
            print(f"  - Test tokens: {len(tokens)} tokens")

        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_sample_file():
    """Test processing a sample from FinPile"""
    print("\n\nTesting sample file processing...")

    input_file = Path(
        "/mnt/z/FinPile/datamix/lda/0fp-100dolma/part-00000-b90aa82b-79fb-41f8-ae2c-3b3e17d3d4ae-c000.json.gz"
    )

    if not input_file.exists():
        print(f"✗ Sample file not found: {input_file}")
        return

    # Use GPT-NeoX tokenizer (no trust_remote_code needed)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Process first 100 documents
    docs_processed = 0
    total_tokens = 0
    start_time = time.time()

    with gzip.open(input_file, "rt") as f:
        for line in f:
            if docs_processed >= 100:
                break

            doc = json.loads(line)
            text = doc.get("text", "")

            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
                docs_processed += 1

    elapsed = time.time() - start_time

    print(f"\n✓ Processed {docs_processed} documents in {elapsed:.2f} seconds")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Avg tokens/doc: {total_tokens / docs_processed:.0f}")
    print(f"  - Processing rate: {docs_processed / elapsed:.0f} docs/sec")

    # Estimate for full dataset
    total_files = 200
    docs_per_file = 285000  # Approximate
    total_docs = total_files * docs_per_file
    est_time_hours = (total_docs / (docs_processed / elapsed)) / 3600

    print("\nEstimates for full dataset:")
    print(f"  - Total documents: ~{total_docs:,}")
    print(f"  - Single-threaded time: ~{est_time_hours:.1f} hours")
    print(f"  - With 4 workers: ~{est_time_hours / 4:.1f} hours")
    print(f"  - With 8 workers: ~{est_time_hours / 8:.1f} hours")


def check_disk_space():
    """Check available disk space"""
    print("\n\nChecking disk space...")

    import shutil

    output_path = Path("data/tokenized/finpile_lda")
    output_path.mkdir(parents=True, exist_ok=True)

    # Get disk usage
    total, used, free = shutil.disk_usage(output_path.parent)

    print(f"Disk space for {output_path.parent}:")
    print(f"  - Total: {total / (1024**3):.1f} GB")
    print(f"  - Used: {used / (1024**3):.1f} GB")
    print(f"  - Free: {free / (1024**3):.1f} GB")

    # Estimate needed space (77GB for full dataset)
    needed_gb = 77
    if free / (1024**3) < needed_gb:
        print("\n⚠️  WARNING: May not have enough space!")
        print(f"  - Needed: ~{needed_gb} GB")
        print(f"  - Available: {free / (1024**3):.1f} GB")
    else:
        print(f"\n✓ Sufficient space available for ~{needed_gb} GB output")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Tokenization Configuration Test")
    print("=" * 60)

    test_tokenizer_setup()
    test_sample_file()
    check_disk_space()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
