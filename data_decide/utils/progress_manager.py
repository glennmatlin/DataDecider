"""Unified progress manager for DataDecider training with Rich terminal UI."""

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


class ProgressManager:
    """Manages all progress displays and status updates for training."""

    def __init__(self, quiet: bool = False, verbose: bool = False):
        """Initialize progress manager.

        Args:
            quiet: Suppress all output
            verbose: Show detailed output
        """
        self.quiet = quiet
        self.verbose = verbose
        self.console = Console()

        # Track active progress bars
        self.progress_bars = {}
        self.metrics = {}
        self.phase = "initializing"
        self.start_time = time.time()

        # Create main progress display
        if not quiet:
            self._setup_progress_display()

    def _setup_progress_display(self):
        """Setup the rich progress display with multiple progress bars."""
        # Define progress bar columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

        # Add standard progress bars
        self.epoch_bar = self.progress.add_task("[cyan]Epochs", total=100, visible=False)
        self.step_bar = self.progress.add_task("[green]Steps", total=100, visible=False)
        self.eval_bar = self.progress.add_task("[yellow]Evaluation", total=100, visible=False)
        self.data_bar = self.progress.add_task("[magenta]Data Loading", total=100, visible=False)

    def set_phase(self, phase: str):
        """Set the current training phase."""
        self.phase = phase
        if not self.quiet:
            phase_colors = {
                "initializing": "blue",
                "warmup": "yellow",
                "training": "green",
                "evaluating": "cyan",
                "saving": "magenta",
                "completed": "green",
                "error": "red",
            }
            color = phase_colors.get(phase, "white")
            self.console.print(f"\n[{color}]▶ {phase.upper()}[/{color}]")

    def update_epoch(self, current: int, total: int):
        """Update epoch progress."""
        if not self.quiet:
            self.progress.update(
                self.epoch_bar,
                completed=current,
                total=total,
                visible=True,
                description=f"[cyan]Epoch {current}/{total}",
            )

    def update_step(self, current: int, total: int, loss: Optional[float] = None):
        """Update step progress."""
        if not self.quiet:
            desc = f"[green]Step {current}/{total}"
            if loss is not None:
                desc += f" | Loss: {loss:.4f}"
            self.progress.update(self.step_bar, completed=current, total=total, visible=True, description=desc)

    def update_eval(self, current: int, total: int):
        """Update evaluation progress."""
        if not self.quiet:
            self.progress.update(
                self.eval_bar,
                completed=current,
                total=total,
                visible=True,
                description=f"[yellow]Evaluating {current}/{total}",
            )

    def update_data_loading(self, current: int, total: int):
        """Update data loading progress."""
        if not self.quiet:
            self.progress.update(
                self.data_bar,
                completed=current,
                total=total,
                visible=True,
                description=f"[magenta]Loading batch {current}/{total}",
            )

    def update_metrics(self, metrics: Dict[str, float]):
        """Update displayed metrics."""
        self.metrics.update(metrics)
        if self.verbose and not self.quiet:
            self._display_metrics_table()

    def _display_metrics_table(self):
        """Display metrics in a formatted table."""
        table = Table(title="Training Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                table.add_row(key, f"{value:.6f}")
            else:
                table.add_row(key, str(value))

        self.console.print(table)

    def log_message(self, message: str, level: str = "info"):
        """Log a message with appropriate styling."""
        if self.quiet:
            return

        level_styles = {
            "info": "[blue]ℹ[/blue]",
            "success": "[green]✓[/green]",
            "warning": "[yellow]⚠[/yellow]",
            "error": "[red]✗[/red]",
            "debug": "[dim]◦[/dim]",
        }

        if level == "debug" and not self.verbose:
            return

        prefix = level_styles.get(level, "")
        self.console.print(f"{prefix} {message}")

    def create_live_display(self) -> Live:
        """Create a live display for real-time metrics."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=6),
            Layout(name="metrics", size=10),
            Layout(name="footer", size=3),
        )

        # Header
        header = Panel(Text("DataDecider Training Monitor", style="bold blue", justify="center"), box=box.DOUBLE)
        layout["header"].update(header)

        # Progress bars
        layout["progress"].update(Panel(self.progress))

        # Metrics will be updated dynamically
        layout["metrics"].update(Panel("Initializing...", title="Metrics"))

        # Footer
        layout["footer"].update(Panel(f"Phase: {self.phase}", style="dim"))

        return Live(layout, console=self.console, refresh_per_second=4)

    @contextmanager
    def progress_context(self):
        """Context manager for progress display."""
        if not self.quiet:
            with self.progress:
                yield self
        else:
            yield self

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time

    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def summary(self):
        """Display training summary."""
        if self.quiet:
            return

        elapsed = self.get_elapsed_time()

        summary_table = Table(title="Training Summary", box=box.DOUBLE)
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Time", self.format_time(elapsed))
        summary_table.add_row("Final Phase", self.phase)

        if self.metrics:
            summary_table.add_row("", "")  # Empty row
            for key, value in sorted(self.metrics.items()):
                if "final" in key.lower() or "best" in key.lower():
                    if isinstance(value, float):
                        summary_table.add_row(key, f"{value:.6f}")
                    else:
                        summary_table.add_row(key, str(value))

        self.console.print("\n")
        self.console.print(summary_table)

    def print_config(self, config: Dict[str, Any], title: str = "Configuration"):
        """Print configuration in a formatted panel."""
        if self.quiet:
            return

        config_text = []
        for key, value in config.items():
            if isinstance(value, dict):
                config_text.append(f"[cyan]{key}:[/cyan]")
                for k, v in value.items():
                    config_text.append(f"  {k}: {v}")
            else:
                config_text.append(f"[cyan]{key}:[/cyan] {value}")

        panel = Panel("\n".join(config_text), title=title, box=box.ROUNDED, expand=False)
        self.console.print(panel)

    def print_system_info(self, info: Dict[str, Any]):
        """Print system information."""
        if self.quiet:
            return

        table = Table(title="System Information", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for key, value in info.items():
            table.add_row(key, str(value))

        self.console.print(table)


# Convenience functions for quick usage
def create_progress_manager(quiet: bool = False, verbose: bool = False) -> ProgressManager:
    """Create a progress manager instance."""
    return ProgressManager(quiet=quiet, verbose=verbose)


def simple_progress_bar(iterable, desc: str = "Processing", quiet: bool = False):
    """Simple progress bar wrapper for iterables."""
    if quiet:
        yield from iterable
    else:
        console = Console()
        with Progress(console=console) as progress:
            task = progress.add_task(f"[cyan]{desc}", total=len(iterable))
            for item in iterable:
                yield item
                progress.advance(task)
