#!/usr/bin/env python3
"""Monitor streaming tokenization progress."""

import json
import time
from pathlib import Path

def monitor():
    output_dir = Path("/mnt/z/FinPile/tokenized/0fp-100dolma_streaming")
    
    while True:
        # Check progress file
        progress_file = output_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                completed = len(progress.get("completed_files", []))
                print(f"\rCompleted: {completed}/200 files ({completed/200*100:.1f}%)", end="", flush=True)
        
        # Check output files
        arrow_files = list(output_dir.glob("tokenized_*.arrow"))
        tmp_files = list(output_dir.glob("tokenized_*.tmp"))
        
        if arrow_files or tmp_files:
            total_size = sum(f.stat().st_size for f in arrow_files + tmp_files)
            size_gb = total_size / 1e9
            print(f" | Output: {len(arrow_files)} files, {size_gb:.1f}GB", end="", flush=True)
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")