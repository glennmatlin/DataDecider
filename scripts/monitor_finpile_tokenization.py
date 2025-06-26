#!/usr/bin/env python3
"""Monitor FinPile tokenization progress in real-time."""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def load_processing_state(state_path: Path) -> dict:
    """Load the current processing state."""
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


def calculate_stats(state: dict, total_files: int = 200) -> dict:
    """Calculate statistics from processing state."""
    if not state:
        return {}
    
    processed_files = len(state.get("processed_files", []))
    failed_files = len(state.get("failed_files", []))
    total_docs = state.get("total_documents", 0)
    total_tokens = state.get("total_tokens", 0)
    total_sequences = state.get("total_sequences", 0)
    
    # Time calculations
    start_time = datetime.fromisoformat(state["start_time"])
    elapsed = datetime.now() - start_time
    
    # Progress
    progress_pct = (processed_files / total_files) * 100 if total_files > 0 else 0
    
    # Estimates
    if processed_files > 0:
        files_per_second = processed_files / elapsed.total_seconds()
        remaining_files = total_files - processed_files
        est_remaining = timedelta(seconds=remaining_files / files_per_second) if files_per_second > 0 else None
        
        # Token rates
        tokens_per_second = total_tokens / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        docs_per_second = total_docs / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
    else:
        est_remaining = None
        tokens_per_second = 0
        docs_per_second = 0
    
    return {
        "processed_files": processed_files,
        "failed_files": failed_files,
        "total_files": total_files,
        "progress_pct": progress_pct,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "total_sequences": total_sequences,
        "elapsed": elapsed,
        "est_remaining": est_remaining,
        "tokens_per_second": tokens_per_second,
        "docs_per_second": docs_per_second,
        "start_time": start_time,
        "last_checkpoint": datetime.fromisoformat(state.get("last_checkpoint", state["start_time"])),
    }


def create_progress_panel(stats: dict) -> Panel:
    """Create progress panel."""
    if not stats:
        return Panel("No processing state found. Waiting for tokenization to start...")
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        expand=True,
    )
    
    task = progress.add_task(
        f"Files: {stats['processed_files']}/{stats['total_files']} ({stats['progress_pct']:.1f}%)",
        total=stats['total_files'],
        completed=stats['processed_files']
    )
    
    return Panel(progress, title="Progress", border_style="green")


def create_stats_table(stats: dict) -> Table:
    """Create statistics table."""
    table = Table(title="Tokenization Statistics", show_header=False)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green")
    
    if stats:
        table.add_row("Start Time", stats['start_time'].strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Elapsed Time", str(stats['elapsed']).split('.')[0])
        table.add_row("Est. Remaining", str(stats['est_remaining']).split('.')[0] if stats['est_remaining'] else "N/A")
        table.add_row("", "")  # Empty row
        table.add_row("Processed Files", f"{stats['processed_files']:,} / {stats['total_files']}")
        table.add_row("Failed Files", f"{stats['failed_files']:,}")
        table.add_row("Progress", f"{stats['progress_pct']:.1f}%")
        table.add_row("", "")  # Empty row
        table.add_row("Total Documents", f"{stats['total_docs']:,}")
        table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        table.add_row("Total Sequences", f"{stats['total_sequences']:,}")
        table.add_row("", "")  # Empty row
        table.add_row("Docs/Second", f"{stats['docs_per_second']:.1f}")
        table.add_row("Tokens/Second", f"{stats['tokens_per_second']:,.0f}")
        table.add_row("Last Checkpoint", stats['last_checkpoint'].strftime("%H:%M:%S"))
    
    return table


def create_checkpoint_info(output_path: Path) -> Table:
    """Create checkpoint information table."""
    table = Table(title="Checkpoints", show_header=True)
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Sequences", style="green")
    table.add_column("Time", style="yellow")
    
    # Find checkpoint directories
    checkpoints = sorted([d for d in output_path.glob("checkpoint_*") if d.is_dir()])
    
    for checkpoint_dir in checkpoints[-5:]:  # Show last 5 checkpoints
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                table.add_row(
                    checkpoint_dir.name,
                    f"{metadata['sequences']:,}",
                    datetime.fromisoformat(metadata['timestamp']).strftime("%H:%M:%S")
                )
    
    return table


def create_layout(stats: dict, output_path: Path) -> Layout:
    """Create the display layout."""
    layout = Layout()
    
    # Main layout
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    
    # Header
    layout["header"].update(
        Panel(
            "[bold cyan]FinPile Tokenization Monitor[/bold cyan]\n"
            "[dim]Press Ctrl+C to exit[/dim]",
            style="white on blue",
        )
    )
    
    # Body split
    layout["body"].split_row(
        Layout(name="stats", ratio=2),
        Layout(name="checkpoints", ratio=1),
    )
    
    # Stats section
    layout["body"]["stats"].split_column(
        Layout(create_progress_panel(stats), size=5),
        Layout(create_stats_table(stats)),
    )
    
    # Checkpoints section
    layout["body"]["checkpoints"].update(create_checkpoint_info(output_path))
    
    # Footer
    if stats and stats['total_tokens'] > 0:
        est_size_gb = stats['total_tokens'] * 2 / 1e9  # Rough estimate
        footer_text = f"Estimated Output Size: {est_size_gb:.1f} GB"
    else:
        footer_text = "Waiting for data..."
    
    layout["footer"].update(
        Panel(footer_text, style="dim")
    )
    
    return layout


def monitor_tokenization(output_path: str, refresh_rate: float = 1.0):
    """Monitor the tokenization progress."""
    output_path = Path(output_path)
    state_path = output_path / "processing_state.json"
    
    console.print("[bold green]Starting FinPile Tokenization Monitor[/bold green]")
    console.print(f"Monitoring: {output_path}")
    console.print(f"State file: {state_path}")
    console.print()
    
    with Live(auto_refresh=False, console=console) as live:
        while True:
            try:
                # Load current state
                state = load_processing_state(state_path)
                
                # Calculate statistics
                stats = calculate_stats(state) if state else {}
                
                # Create and update layout
                layout = create_layout(stats, output_path)
                live.update(layout)
                live.refresh()
                
                # Check if processing is complete
                if stats and stats['progress_pct'] >= 100:
                    console.print("\n[bold green]Tokenization Complete![/bold green]")
                    break
                
                # Wait before next update
                time.sleep(refresh_rate)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                time.sleep(5)  # Wait longer on error


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor FinPile tokenization progress")
    parser.add_argument(
        "--output-path",
        type=str,
        default="/mnt/z/FinPile/tokenized/0fp-100dolma",
        help="Output directory being monitored"
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=1.0,
        help="Refresh rate in seconds"
    )
    
    args = parser.parse_args()
    
    try:
        monitor_tokenization(args.output_path, args.refresh_rate)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()