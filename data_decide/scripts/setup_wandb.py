#!/usr/bin/env python3
"""Setup script for Weights & Biases configuration."""

from pathlib import Path


def setup_wandb():
    """Interactive setup for W&B configuration."""
    print("Weights & Biases Setup for OLMo Training")
    print("=" * 50)

    # Check if .env file exists
    env_file = Path(".env")

    if env_file.exists():
        print("Found existing .env file")
        overwrite = input("Overwrite existing configuration? (y/N): ").lower()
        if overwrite != "y":
            print("Keeping existing configuration")
            return

    print("\nTo use W&B, you need an API key from https://wandb.ai/")
    print("1. Sign up or log in at https://wandb.ai/")
    print("2. Go to your settings: https://wandb.ai/settings")
    print("3. Copy your API key")
    print()

    # Get API key
    api_key = input("Enter your W&B API key (or press Enter to skip): ").strip()

    if not api_key:
        print("\nSkipping W&B setup. You can run this script again later.")
        return

    # Get optional configurations
    print("\nOptional configurations (press Enter to use defaults):")

    entity = input("W&B Entity/Username (default: your username): ").strip()
    project = input("W&B Project name (default: olmo-4m-datadecide): ").strip() or "olmo-4m-datadecide"

    # Create .env file
    env_content = f"""# Weights & Biases Configuration
WANDB_API_KEY={api_key}
WANDB_PROJECT={project}
"""

    if entity:
        env_content += f"WANDB_ENTITY={entity}\n"

    # Additional optional settings
    env_content += """
# Optional: Disable W&B if needed
# WANDB_MODE=offline

# Optional: Custom W&B directory
# WANDB_DIR=./wandb

# Optional: Silence W&B console output
# WANDB_SILENT=true
"""

    # Write .env file
    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"\n✅ Created {env_file}")

    # Test W&B connection
    print("\nTesting W&B connection...")
    try:
        import wandb

        wandb.login(key=api_key)
        print("✅ Successfully connected to W&B!")

        # Create a test run
        test = input("\nCreate a test run to verify setup? (Y/n): ").lower()
        if test != "n":
            run = wandb.init(
                project=project,
                entity=entity if entity else None,
                name="test-connection",
                tags=["test"],
                notes="Testing W&B connection",
            )

            # Log a test metric
            wandb.log({"test_metric": 1.0})

            # Finish run
            wandb.finish()

            print(f"✅ Test run created! View at: {run.url}")

    except ImportError:
        print("⚠️  W&B not installed. Install with: pip install wandb")
    except Exception as e:
        print(f"❌ Error connecting to W&B: {e}")

    print("\nSetup complete! Your training will now log to W&B.")
    print("\nUseful W&B commands:")
    print("  wandb login          # Log in to W&B")
    print("  wandb offline        # Run in offline mode")
    print("  wandb online         # Run in online mode")
    print("  wandb sync           # Sync offline runs")


if __name__ == "__main__":
    setup_wandb()
