#!/usr/bin/env python3
"""Live monitoring dashboard for OLMo training."""

import os
import sys
import time
import wandb
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.plot import Plot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

def create_layout():
    """Create the monitoring dashboard layout."""
    layout = Layout()
    
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=2),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="progress", ratio=1)
    )
    
    return layout

def get_header():
    """Create header panel."""
    return Panel(
        "[bold blue]OLMo 4M Training Monitor[/bold blue]\n"
        f"[dim]Live dashboard • Updated: {datetime.now().strftime('%H:%M:%S')}[/dim]",
        border_style="blue"
    )

def get_metrics_table(run_data):
    """Create metrics table from W&B data."""
    table = Table(title="Current Metrics", expand=True)
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Best", style="green")
    
    # Extract latest values
    history = run_data.history()
    if not history.empty:
        latest = history.iloc[-1]
        
        # Training metrics
        if 'train/loss' in latest:
            best_loss = history['train/loss'].min()
            table.add_row("Loss", f"{latest['train/loss']:.4f}", f"{best_loss:.4f}")
        
        if 'train/perplexity' in latest:
            best_ppl = history['train/perplexity'].min()
            table.add_row("Perplexity", f"{latest['train/perplexity']:.2f}", f"{best_ppl:.2f}")
        
        if 'train/learning_rate' in latest:
            table.add_row("Learning Rate", f"{latest['train/learning_rate']:.2e}", "—")
        
        if 'train/tokens_per_sec' in latest:
            avg_tps = history['train/tokens_per_sec'].mean()
            table.add_row("Tokens/sec", f"{latest['train/tokens_per_sec']:,.0f}", f"{avg_tps:,.0f} avg")
        
        # DataDecide metrics
        if 'datadecide/combined_score' in latest:
            best_score = history['datadecide/combined_score'].max()
            table.add_row("DataDecide Score", f"{latest['datadecide/combined_score']:.2f}", f"{best_score:.2f}")
        
        # Evaluation metrics
        if 'eval/loss' in latest:
            best_eval = history['eval/loss'].min()
            table.add_row("Eval Loss", f"{latest['eval/loss']:.4f}", f"{best_eval:.4f}")
    
    return table

def get_progress_panel(run_data):
    """Create progress panel."""
    config = run_data.config
    summary = run_data.summary
    
    # Get current step
    history = run_data.history()
    current_step = 0
    if not history.empty and 'train/global_step' in history.columns:
        current_step = int(history['train/global_step'].iloc[-1])
    
    total_steps = config.get('training_steps', 5725)
    progress_pct = (current_step / total_steps) * 100
    
    # Create progress bar
    progress = Progress(
        TextColumn("[bold blue]Training Progress"),
        BarColumn(bar_width=40),
        TextColumn(f"{current_step:,}/{total_steps:,} steps"),
        TextColumn(f"({progress_pct:.1f}%)"),
        expand=True
    )
    
    task = progress.add_task("", total=total_steps, completed=current_step)
    
    # Time estimates
    start_time = datetime.fromisoformat(run_data.metadata['startedAt'].replace('Z', '+00:00'))
    elapsed = datetime.now() - start_time.replace(tzinfo=None)
    
    if current_step > 0:
        time_per_step = elapsed.total_seconds() / current_step
        remaining_steps = total_steps - current_step
        eta = time_per_step * remaining_steps
        eta_str = f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
    else:
        eta_str = "Calculating..."
    
    info = f"\n[dim]Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n"
    info += f"[dim]Elapsed: {str(elapsed).split('.')[0]}[/dim]\n"
    info += f"[dim]ETA: {eta_str}[/dim]"
    
    return Panel(
        str(progress) + info,
        title="Progress",
        border_style="green"
    )

def get_footer(run_data):
    """Create footer with run info."""
    return Panel(
        f"[bold]Run:[/bold] {run_data.name} • "
        f"[bold]Project:[/bold] {run_data.project} • "
        f"[bold]State:[/bold] {run_data.state} • "
        f"[link={run_data.url}]Open in W&B[/link]",
        border_style="dim"
    )

def monitor_run(project_name="olmo-4m-datadecide", run_name=None):
    """Monitor a W&B run with live updates."""
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Get the run
    if run_name:
        run_path = f"{project_name}/{run_name}"
    else:
        # Get the latest run
        runs = api.runs(project_name, order="-created_at", per_page=1)
        if not runs:
            console.print("[red]No runs found in project![/red]")
            return
        run = runs[0]
        run_path = f"{run.entity}/{run.project}/{run.id}"
    
    console.print(f"[green]Monitoring run:[/green] {run_path}")
    
    layout = create_layout()
    
    with Live(layout, refresh_per_second=0.5, console=console) as live:
        while True:
            try:
                # Refresh run data
                run = api.run(run_path)
                
                # Update layout
                layout["header"].update(get_header())
                layout["metrics"].update(get_metrics_table(run))
                layout["progress"].update(get_progress_panel(run))
                layout["footer"].update(get_footer(run))
                
                # Check if run is finished
                if run.state in ["finished", "failed", "crashed"]:
                    console.print(f"\n[bold]Run {run.state}![/bold]")
                    break
                
                # Wait before next update
                time.sleep(5)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                time.sleep(10)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor OLMo training runs")
    parser.add_argument("--project", default="olmo-4m-datadecide", help="W&B project name")
    parser.add_argument("--run", default=None, help="Specific run name to monitor")
    
    args = parser.parse_args()
    
    monitor_run(args.project, args.run)

if __name__ == "__main__":
    main()