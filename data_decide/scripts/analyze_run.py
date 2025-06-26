#!/usr/bin/env python3
"""Analyze completed W&B runs and generate reports."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load environment variables
load_dotenv()

console = Console()


def analyze_run(run_path: str, save_plots: bool = True):
    """Analyze a completed W&B run."""

    api = wandb.Api()
    run = api.run(run_path)

    console.print(
        Panel.fit(
            f"[bold blue]W&B Run Analysis[/bold blue]\nRun: {run.name}\nProject: {run.project}\nState: {run.state}",
            border_style="blue",
        )
    )

    # Get run history
    history = run.history()
    config = run.config
    summary = run.summary

    # Training Summary
    console.print("\n[bold]Training Summary[/bold]")
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")

    summary_table.add_row("Total Steps", f"{summary.get('training_steps', 'N/A'):,}")
    summary_table.add_row("Final Loss", f"{summary.get('final_loss', 'N/A'):.4f}" if "final_loss" in summary else "N/A")
    summary_table.add_row("Best Eval Loss", f"{history['eval/loss'].min():.4f}" if "eval/loss" in history else "N/A")
    summary_table.add_row(
        "Best Perplexity", f"{history['eval/perplexity'].min():.2f}" if "eval/perplexity" in history else "N/A"
    )
    summary_table.add_row("Training Time", summary.get("training_time", "N/A"))
    summary_table.add_row("Total Tokens", f"{summary.get('dataset_tokens', 0):,}")

    console.print(summary_table)

    # DataDecide Metrics Analysis
    if any("datadecide" in col for col in history.columns):
        console.print("\n[bold]DataDecide Metrics Analysis[/bold]")
        dd_table = Table()
        dd_table.add_column("Metric", style="cyan")
        dd_table.add_column("Mean", style="magenta")
        dd_table.add_column("Std", style="blue")
        dd_table.add_column("Min", style="red")
        dd_table.add_column("Max", style="green")

        for metric in ["perplexity", "diversity", "quality", "combined_score"]:
            col_name = f"datadecide/{metric}"
            if col_name in history:
                values = history[col_name].dropna()
                dd_table.add_row(
                    metric.capitalize(),
                    f"{values.mean():.3f}",
                    f"{values.std():.3f}",
                    f"{values.min():.3f}",
                    f"{values.max():.3f}",
                )

        console.print(dd_table)

    # Training Efficiency
    console.print("\n[bold]Training Efficiency[/bold]")
    if "train/tokens_per_sec" in history:
        tokens_per_sec = history["train/tokens_per_sec"].dropna()
        efficiency_table = Table(show_header=False)
        efficiency_table.add_column("Metric", style="cyan")
        efficiency_table.add_column("Value", style="yellow")

        efficiency_table.add_row("Avg Tokens/sec", f"{tokens_per_sec.mean():,.0f}")
        efficiency_table.add_row("Max Tokens/sec", f"{tokens_per_sec.max():,.0f}")
        efficiency_table.add_row("Min Tokens/sec", f"{tokens_per_sec.min():,.0f}")

        if "train/gpu_memory_gb" in history:
            gpu_mem = history["train/gpu_memory_gb"].dropna()
            efficiency_table.add_row("Avg GPU Memory", f"{gpu_mem.mean():.2f} GB")
            efficiency_table.add_row("Peak GPU Memory", f"{gpu_mem.max():.2f} GB")

        console.print(efficiency_table)

    # Generate plots
    if save_plots:
        output_dir = Path(f"analysis_{run.name}")
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]Generating plots in {output_dir}/[/bold]")

        # Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Training Analysis: {run.name}")

        # Training loss
        if "train/loss" in history:
            ax = axes[0, 0]
            history["train/loss"].dropna().plot(ax=ax)
            ax.set_title("Training Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True)

        # Eval metrics
        if "eval/loss" in history and "eval/perplexity" in history:
            ax = axes[0, 1]
            ax2 = ax.twinx()

            eval_steps = history[history["eval/loss"].notna()].index
            ax.plot(eval_steps, history.loc[eval_steps, "eval/loss"], "b-", label="Loss")
            ax2.plot(eval_steps, history.loc[eval_steps, "eval/perplexity"], "r-", label="Perplexity")

            ax.set_xlabel("Step")
            ax.set_ylabel("Loss", color="b")
            ax2.set_ylabel("Perplexity", color="r")
            ax.set_title("Evaluation Metrics")
            ax.grid(True)

        # Learning rate
        if "train/learning_rate" in history:
            ax = axes[1, 0]
            history["train/learning_rate"].dropna().plot(ax=ax)
            ax.set_title("Learning Rate Schedule")
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.grid(True)

        # DataDecide combined score
        if "datadecide/combined_score" in history:
            ax = axes[1, 1]
            history["datadecide/combined_score"].dropna().plot(ax=ax, color="green")
            ax.set_title("DataDecide Combined Score")
            ax.set_xlabel("Step")
            ax.set_ylabel("Score")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / "training_analysis.png", dpi=150, bbox_inches="tight")
        console.print("[green]✓[/green] Saved training_analysis.png")

        # Additional DataDecide metrics plot
        if any("datadecide" in col for col in history.columns):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("DataDecide Metrics Evolution")

            metrics = ["perplexity", "diversity", "quality", "combined_score"]
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                col_name = f"datadecide/{metric}"
                if col_name in history:
                    history[col_name].dropna().plot(ax=ax)
                    ax.set_title(metric.capitalize())
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Value")
                    ax.grid(True)

            plt.tight_layout()
            plt.savefig(output_dir / "datadecide_metrics.png", dpi=150, bbox_inches="tight")
            console.print("[green]✓[/green] Saved datadecide_metrics.png")

    # Export data
    console.print("\n[bold]Exporting data...[/bold]")

    # Save configuration
    config_path = output_dir / "config.json" if save_plots else "config.json"
    import json

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"[green]✓[/green] Saved configuration to {config_path}")

    # Save history
    history_path = output_dir / "history.csv" if save_plots else "history.csv"
    history.to_csv(history_path)
    console.print(f"[green]✓[/green] Saved history to {history_path}")

    # Generate final report
    report_path = output_dir / "report.txt" if save_plots else "report.txt"
    with open(report_path, "w") as f:
        f.write(f"Training Report for {run.name}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Project: {run.project}\n")
        f.write(f"Run ID: {run.id}\n")
        f.write(f"State: {run.state}\n")
        f.write(f"Start Time: {run.metadata.get('startedAt', 'N/A')}\n")
        f.write(f"URL: {run.url}\n\n")

        f.write("Configuration:\n")
        f.write("-" * 30 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\nFinal Metrics:\n")
        f.write("-" * 30 + "\n")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    console.print(f"[green]✓[/green] Saved report to {report_path}")

    console.print(
        Panel.fit(
            f"[bold green]Analysis Complete![/bold green]\nFiles saved to: {output_dir if save_plots else '.'}/",
            border_style="green",
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B training runs")
    parser.add_argument("run", help="W&B run path (e.g., username/project/run_id)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    try:
        analyze_run(args.run, save_plots=not args.no_plots)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\nMake sure the run path is in the format: username/project/run_id")


if __name__ == "__main__":
    main()
