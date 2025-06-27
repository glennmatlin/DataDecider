#!/usr/bin/env python3
"""Quick script to check FinPile tokenization progress."""

import json
import os
from datetime import datetime
from pathlib import Path


def check_progress():
    output_path = Path("/mnt/z/FinPile/tokenized/0fp-100dolma")
    state_file = output_path / "processing_state.json"

    print(f"FinPile Tokenization Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check if process is running
    ps_output = os.popen("ps aux | grep tokenize_finpile_enhanced | grep -v grep").read()
    if ps_output:
        print("✓ Tokenization process is running")
    else:
        print("✗ Tokenization process not found")

    # Check state file
    if state_file.exists():
        with open(state_file, "r") as f:
            state = json.load(f)

        processed = len(state.get("processed_files", []))
        total = 200
        progress_pct = (processed / total) * 100

        print(f"\nFiles processed: {processed}/{total} ({progress_pct:.1f}%)")
        print(f"Total documents: {state.get('total_documents', 0):,}")
        print(f"Total tokens: {state.get('total_tokens', 0):,}")
        print(f"Total sequences: {state.get('total_sequences', 0):,}")
        print(f"Failed files: {len(state.get('failed_files', []))}")
        print(f"Errors: {state.get('error_count', 0)}")

        # Time estimate
        start_time = datetime.fromisoformat(state["start_time"])
        elapsed = datetime.now() - start_time
        if processed > 0:
            est_total = elapsed * total / processed
            est_remaining = est_total - elapsed
            print(f"\nElapsed time: {elapsed}")
            print(f"Estimated remaining: {est_remaining}")
    else:
        print("\nNo state file found. Tokenization may not have started saving progress yet.")

    # Check checkpoints
    checkpoints = list(output_path.glob("checkpoint_*"))
    if checkpoints:
        print(f"\nCheckpoints saved: {len(checkpoints)}")
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Latest checkpoint: {latest.name}")

    # Check disk usage
    if output_path.exists():
        size_output = os.popen(f"du -sh {output_path}").read().strip()
        if size_output:
            size = size_output.split()[0]
            print(f"\nOutput size so far: {size}")

    print("=" * 60)


if __name__ == "__main__":
    check_progress()
