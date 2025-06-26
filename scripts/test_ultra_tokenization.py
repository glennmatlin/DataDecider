#!/usr/bin/env python3
"""Test ultra-fast tokenization on a subset of files."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tokenize_finpile_ultra import UltraConfig, UltraTokenizer


def main():
    # Test configuration - process only first 5 files
    config = UltraConfig(
        input_dir="/mnt/z/FinPile/0fp-100dolma",
        output_dir="/mnt/z/FinPile/tokenized/ultra_test",
        tokenizer_name="EleutherAI/gpt-neox-20b",
        max_seq_length=2048,
        batch_size=100,
        sequences_per_chunk=50000,
        num_workers=8,  # Use 8 parallel workers
        prefetch_files=4,
    )
    
    # Limit to first 5 files for testing
    input_dir = Path(config.input_dir)
    test_files = sorted(input_dir.glob("*.json.gz"))[:5]
    
    # Create test directory with symlinks
    test_input_dir = Path("/tmp/finpile_test_input")
    test_input_dir.mkdir(exist_ok=True)
    
    for f in test_files:
        link = test_input_dir / f.name
        if not link.exists():
            link.symlink_to(f)
    
    # Update config to use test directory
    config.input_dir = str(test_input_dir)
    
    print("Ultra-fast tokenization test")
    print(f"Processing {len(test_files)} files with {config.num_workers} workers")
    print(f"Output: {config.output_dir}")
    
    # Run tokenization
    tokenizer = UltraTokenizer(config)
    tokenizer.run()
    
    # Cleanup
    import shutil
    shutil.rmtree(test_input_dir)


if __name__ == "__main__":
    main()