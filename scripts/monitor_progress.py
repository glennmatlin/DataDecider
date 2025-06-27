#!/usr/bin/env python3
"""
Universal progress monitor for all tokenization formats.
Works with unified tokenizer, hybrid, streaming, and legacy formats.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

import pyarrow as pa
import pyarrow.parquet as pq


class UniversalProgressMonitor:
    """Monitor tokenization progress across different formats and implementations"""

    def __init__(self, output_dir: str, checkpoint_dir: str = None, mode: str = "auto"):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.mode = mode
        self.start_time = time.time()

    def detect_format(self) -> str:
        """Auto-detect which tokenization format is being used"""
        if self.mode != "auto":
            return self.mode

        # Check for unified tokenizer checkpoint
        if self.checkpoint_dir and (self.checkpoint_dir / "progress.json").exists():
            return "unified"

        # Check for streaming tokenizer files
        if (self.output_dir / "progress.json").exists():
            return "streaming"

        # Check for enhanced/old format state file
        if (self.output_dir / "processing_state.json").exists():
            return "enhanced"

        # Check for hybrid format (just parquet files)
        if list(self.output_dir.glob("*.parquet")):
            return "hybrid"

        # Check for arrow format
        if list(self.output_dir.glob("*.arrow")):
            return "arrow"

        return "unknown"

    def get_process_info(self) -> Dict:
        """Get information about running tokenization processes"""
        info = {"processes": [], "total_cpu": 0.0, "total_memory_gb": 0.0}

        # Check for running processes
        process_patterns = [
            "tokenize_unified",
            "tokenize_finpile_hybrid",
            "tokenize_finpile_streaming",
            "tokenize_datasets",
        ]

        for pattern in process_patterns:
            cmd = f"ps aux | grep {pattern} | grep -v grep"
            output = os.popen(cmd).read().strip()
            if output:
                lines = output.split("\n")
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 11:
                        info["processes"].append(
                            {
                                "name": pattern,
                                "pid": parts[1],
                                "cpu": float(parts[2]),
                                "memory": float(parts[3]),
                                "time": parts[9],
                                "command": " ".join(parts[10:])[:50] + "...",
                            }
                        )
                        info["total_cpu"] += float(parts[2])
                        info["total_memory_gb"] += float(parts[3]) * self._get_total_memory_gb() / 100

        return info

    def _get_total_memory_gb(self) -> float:
        """Get total system memory in GB"""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 / 1024
        except:
            pass
        return 64.0  # Default assumption

    def get_unified_progress(self) -> Dict:
        """Get progress from unified tokenizer format"""
        progress = {
            "format": "unified",
            "status": "unknown",
            "files_processed": 0,
            "total_files": 0,
            "documents": 0,
            "tokens": 0,
            "sequences": 0,
            "errors": 0,
            "elapsed_time": 0,
            "checkpoint_time": None,
        }

        # Read checkpoint
        checkpoint_file = self.checkpoint_dir / "progress.json" if self.checkpoint_dir else None
        if checkpoint_file and checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)

            stats = checkpoint.get("stats", {})
            progress.update(
                {
                    "status": "running",
                    "files_processed": len(checkpoint.get("completed_files", [])),
                    "documents": stats.get("total_documents", 0),
                    "tokens": stats.get("total_tokens", 0),
                    "sequences": stats.get("total_sequences", 0),
                    "errors": stats.get("errors", 0),
                    "elapsed_time": stats.get("elapsed_time", 0),
                    "checkpoint_time": checkpoint.get("timestamp"),
                }
            )

        # Count output files
        progress["output_files"] = len(list(self.output_dir.glob("tokenized_*.arrow"))) + len(
            list(self.output_dir.glob("tokenized_*.parquet"))
        )

        return progress

    def get_streaming_progress(self) -> Dict:
        """Get progress from streaming tokenizer format"""
        progress = {
            "format": "streaming",
            "status": "unknown",
            "files_processed": 0,
            "total_files": 200,  # Hardcoded for FinPile
            "documents": 0,
            "tokens": 0,
            "sequences": 0,
        }

        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                data = json.load(f)

            progress.update(
                {
                    "status": "running",
                    "files_processed": len(data.get("completed_files", [])),
                    "checkpoint_time": data.get("last_update"),
                }
            )

        # Count sequences from output files
        total_sequences = 0
        for arrow_file in self.output_dir.glob("*.arrow"):
            try:
                with pa.memory_map(str(arrow_file)) as source:
                    reader = pa.ipc.open_file(source)
                    total_sequences += reader.num_record_batches
            except:
                pass

        progress["sequences"] = total_sequences
        progress["output_files"] = len(list(self.output_dir.glob("*.arrow")))

        return progress

    def get_enhanced_progress(self) -> Dict:
        """Get progress from enhanced/legacy format"""
        progress = {
            "format": "enhanced",
            "status": "unknown",
            "files_processed": 0,
            "total_files": 200,
            "documents": 0,
            "tokens": 0,
            "sequences": 0,
            "errors": 0,
        }

        state_file = self.output_dir / "processing_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            progress.update(
                {
                    "status": "running",
                    "files_processed": len(state.get("processed_files", [])),
                    "documents": state.get("total_documents", 0),
                    "tokens": state.get("total_tokens", 0),
                    "sequences": state.get("total_sequences", 0),
                    "errors": state.get("error_count", 0),
                    "start_time": state.get("start_time"),
                }
            )

        return progress

    def get_output_stats(self) -> Dict:
        """Get statistics from output files"""
        stats = {
            "arrow_files": 0,
            "parquet_files": 0,
            "total_sequences": 0,
            "total_size_gb": 0,
            "avg_sequence_length": 0,
        }

        # Count Arrow files
        sequence_lengths = []
        for arrow_file in self.output_dir.glob("*.arrow"):
            stats["arrow_files"] += 1
            stats["total_size_gb"] += arrow_file.stat().st_size / 1024**3

            try:
                table = pa.ipc.read_file(arrow_file)
                stats["total_sequences"] += len(table)
                if "length" in table.column_names:
                    sequence_lengths.extend(table["length"].to_pylist())
            except:
                pass

        # Count Parquet files
        for parquet_file in self.output_dir.glob("*.parquet"):
            stats["parquet_files"] += 1
            stats["total_size_gb"] += parquet_file.stat().st_size / 1024**3

            try:
                table = pq.read_table(parquet_file)
                stats["total_sequences"] += len(table)
                if "length" in table.column_names:
                    sequence_lengths.extend(table["length"].to_pylist())
            except:
                pass

        if sequence_lengths:
            stats["avg_sequence_length"] = sum(sequence_lengths) / len(sequence_lengths)

        return stats

    def create_display(self, continuous: bool = False) -> Optional[Layout]:
        """Create display output"""
        # Detect format
        format_type = self.detect_format()

        # Get progress based on format
        if format_type == "unified":
            progress = self.get_unified_progress()
        elif format_type == "streaming":
            progress = self.get_streaming_progress()
        elif format_type == "enhanced":
            progress = self.get_enhanced_progress()
        else:
            progress = {"format": "unknown", "status": "no data"}

        # Get process info
        process_info = self.get_process_info()

        # Get output stats
        output_stats = self.get_output_stats()

        if RICH_AVAILABLE and continuous:
            return self._create_rich_display(progress, process_info, output_stats)
        else:
            self._print_text_display(progress, process_info, output_stats)
            return None

    def _create_rich_display(self, progress: Dict, process_info: Dict, output_stats: Dict) -> Layout:
        """Create Rich display layout"""
        layout = Layout()

        # Header
        header = Panel(f"📊 Tokenization Monitor - {progress['format'].upper()} Format", style="bold blue")

        # Progress info
        progress_table = Table(title="Progress", show_header=False, box=None)
        progress_table.add_column("Metric", style="cyan")
        progress_table.add_column("Value", style="green")

        if progress.get("total_files"):
            pct = (progress["files_processed"] / progress["total_files"]) * 100
            progress_table.add_row("Files", f"{progress['files_processed']}/{progress['total_files']} ({pct:.1f}%)")
        else:
            progress_table.add_row("Files", f"{progress['files_processed']}")

        if progress.get("documents"):
            progress_table.add_row("Documents", f"{progress['documents']:,}")
        if progress.get("tokens"):
            progress_table.add_row("Tokens", f"{progress['tokens']:,}")
        if progress.get("sequences"):
            progress_table.add_row("Sequences", f"{progress['sequences']:,}")
        if progress.get("errors"):
            progress_table.add_row("Errors", f"{progress['errors']:,}")

        # Process info
        process_table = Table(title="Running Processes", show_header=True)
        process_table.add_column("Process", style="cyan")
        process_table.add_column("PID", style="yellow")
        process_table.add_column("CPU %", style="green")
        process_table.add_column("Memory", style="blue")

        if process_info["processes"]:
            for proc in process_info["processes"]:
                process_table.add_row(proc["name"], proc["pid"], f"{proc['cpu']:.1f}", f"{proc['memory']:.1f}%")
        else:
            process_table.add_row("No processes found", "", "", "")

        # Output stats
        output_table = Table(title="Output Statistics", show_header=False, box=None)
        output_table.add_column("Metric", style="cyan")
        output_table.add_column("Value", style="magenta")

        output_table.add_row("Arrow Files", str(output_stats["arrow_files"]))
        output_table.add_row("Parquet Files", str(output_stats["parquet_files"]))
        output_table.add_row("Total Sequences", f"{output_stats['total_sequences']:,}")
        output_table.add_row("Disk Size", f"{output_stats['total_size_gb']:.2f} GB")
        if output_stats["avg_sequence_length"] > 0:
            output_table.add_row("Avg Seq Length", f"{output_stats['avg_sequence_length']:.0f}")

        # Time info
        if progress.get("checkpoint_time"):
            try:
                checkpoint_time = datetime.fromisoformat(progress["checkpoint_time"])
                time_ago = datetime.now() - checkpoint_time
                time_text = f"Last update: {time_ago.seconds}s ago"
            except:
                time_text = "Last update: Unknown"
        else:
            time_text = "No checkpoint data"

        time_panel = Panel(time_text, style="dim")

        # Layout assembly
        layout.split_column(
            header,
            Layout(name="main").split_row(Panel(progress_table), Panel(output_table)),
            Panel(process_table),
            time_panel,
        )

        return layout

    def _print_text_display(self, progress: Dict, process_info: Dict, output_stats: Dict):
        """Print text-based display"""
        print(f"\nTokenization Progress Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        print(f"\nFormat: {progress['format'].upper()}")
        print(f"Status: {progress['status']}")

        # Progress
        print("\nProgress:")
        if progress.get("total_files"):
            pct = (progress["files_processed"] / progress["total_files"]) * 100
            print(f"  Files: {progress['files_processed']}/{progress['total_files']} ({pct:.1f}%)")
        else:
            print(f"  Files: {progress['files_processed']}")

        if progress.get("documents"):
            print(f"  Documents: {progress['documents']:,}")
        if progress.get("tokens"):
            print(f"  Tokens: {progress['tokens']:,}")
        if progress.get("sequences"):
            print(f"  Sequences: {progress['sequences']:,}")
        if progress.get("errors"):
            print(f"  Errors: {progress['errors']:,}")

        # Processes
        print(f"\nRunning Processes: {len(process_info['processes'])}")
        if process_info["processes"]:
            print(f"  Total CPU: {process_info['total_cpu']:.1f}%")
            print(f"  Total Memory: {process_info['total_memory_gb']:.2f} GB")

        # Output
        print("\nOutput Statistics:")
        print(f"  Arrow files: {output_stats['arrow_files']}")
        print(f"  Parquet files: {output_stats['parquet_files']}")
        print(f"  Total sequences: {output_stats['total_sequences']:,}")
        print(f"  Disk size: {output_stats['total_size_gb']:.2f} GB")
        if output_stats["avg_sequence_length"] > 0:
            print(f"  Avg sequence length: {output_stats['avg_sequence_length']:.0f}")

        print("=" * 70)

    def run(self, interval: int = 5, continuous: bool = False):
        """Run the monitor"""
        if continuous and RICH_AVAILABLE:
            # Continuous monitoring with Rich
            try:
                with Live(self.create_display(continuous=True), refresh_per_second=0.5) as live:
                    while True:
                        time.sleep(interval)
                        live.update(self.create_display(continuous=True))
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")
        else:
            # Single shot or continuous text mode
            while True:
                self.create_display(continuous=False)

                if not continuous:
                    break

                try:
                    time.sleep(interval)
                except KeyboardInterrupt:
                    print("\nMonitoring stopped by user")
                    break


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Universal progress monitor for tokenization tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("output_dir", type=str, help="Tokenization output directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory (for unified tokenizer)")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "unified", "streaming", "enhanced", "hybrid"],
        help="Tokenization format mode",
    )
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous monitoring mode")
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich UI even if available")

    args = parser.parse_args()

    # Disable Rich if requested
    if args.no_rich:
        global RICH_AVAILABLE
        RICH_AVAILABLE = False

    # Create and run monitor
    monitor = UniversalProgressMonitor(output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir, mode=args.mode)

    monitor.run(interval=args.interval, continuous=args.continuous)


if __name__ == "__main__":
    main()
