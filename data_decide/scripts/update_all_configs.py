#!/usr/bin/env python3
"""Update all OLMo configurations to match the paper exactly."""

from pathlib import Path

import yaml

# Exact hyperparameters from the OLMo paper
OLMO_CONFIGS = {
    "4M": {
        "actual_params": 3.7e6,
        "batch_size": 32,
        "hidden_size": 64,
        "learning_rate": 1.4e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 5725,
        "tokens_trained": 0.4e9,
    },
    "6M": {
        "actual_params": 6.0e6,
        "batch_size": 32,
        "hidden_size": 96,
        "learning_rate": 1.2e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 9182,
        "tokens_trained": 0.6e9,
    },
    "8M": {
        "actual_params": 8.5e6,
        "batch_size": 32,
        "hidden_size": 128,
        "learning_rate": 1.1e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 13039,
        "tokens_trained": 0.9e9,
    },
    "10M": {
        "actual_params": 9.9e6,
        "batch_size": 32,
        "hidden_size": 144,
        "learning_rate": 1.0e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 15117,
        "tokens_trained": 1.0e9,
    },
    "14M": {
        "actual_params": 14.4e6,
        "batch_size": 32,
        "hidden_size": 192,
        "learning_rate": 9.2e-03,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 21953,
        "tokens_trained": 1.4e9,
    },
    "16M": {
        "actual_params": 16.0e6,
        "batch_size": 32,
        "hidden_size": 208,
        "learning_rate": 8.9e-03,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 24432,
        "tokens_trained": 1.6e9,
    },
    "20M": {
        "actual_params": 19.1e6,
        "batch_size": 64,
        "hidden_size": 192,
        "learning_rate": 8.4e-03,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 14584,
        "tokens_trained": 1.9e9,
    },
    "60M": {
        "actual_params": 57.1e6,
        "batch_size": 96,
        "hidden_size": 384,
        "learning_rate": 5.8e-03,
        "num_heads": 12,
        "num_layers": 16,
        "training_steps": 29042,
        "tokens_trained": 5.7e9,
    },
    "90M": {
        "actual_params": 97.9e6,
        "batch_size": 160,
        "hidden_size": 528,
        "learning_rate": 4.9e-03,
        "num_heads": 12,
        "num_layers": 16,
        "training_steps": 29901,
        "tokens_trained": 9.8e9,
    },
    "150M": {
        "actual_params": 151.9e6,
        "batch_size": 192,
        "hidden_size": 768,
        "learning_rate": 4.2e-03,
        "num_heads": 12,
        "num_layers": 12,
        "training_steps": 38157,
        "tokens_trained": 15.0e9,
    },
    "300M": {
        "actual_params": 320.0e6,
        "batch_size": 320,
        "hidden_size": 1024,
        "learning_rate": 3.3e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 45787,
        "tokens_trained": 30.0e9,
    },
    "530M": {
        "actual_params": 530.1e6,
        "batch_size": 448,
        "hidden_size": 1344,
        "learning_rate": 2.8e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 57786,
        "tokens_trained": 53.0e9,
    },
    "750M": {
        "actual_params": 681.3e6,  # Note: actual is less than name
        "batch_size": 576,
        "hidden_size": 1536,
        "learning_rate": 2.5e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 63589,
        "tokens_trained": 75.0e9,
    },
    "1B": {
        "actual_params": 1176.8e6,
        "batch_size": 704,
        "hidden_size": 2048,
        "learning_rate": 2.1e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 69369,
        "tokens_trained": 100.0e9,
    },
}


def create_model_config(model_name: str, params: dict) -> dict:
    """Create model configuration matching OLMo paper."""

    # Calculate intermediate size (usually 4x hidden for SwiGLU)
    # For SwiGLU: intermediate = hidden * 8/3, but rounded to nice numbers
    intermediate_size = int(params["hidden_size"] * 4)

    config = {
        "model": {
            "name": f"olmo-{model_name.lower()}",
            "vocab_size": 50257,  # GPT-NeoX-20B tokenizer
            "hidden_size": params["hidden_size"],
            "num_hidden_layers": params["num_layers"],
            "num_attention_heads": params["num_heads"],
            "intermediate_size": intermediate_size,
            "hidden_act": "swiglu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 2048,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "use_cache": True,
            "rope_theta": 10000.0,
            "use_bias": False,
            "tie_word_embeddings": True,
        },
        "training_params": {
            "actual_model_params": f"{params['actual_params'] / 1e6:.1f}M",
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "training_steps": params["training_steps"],
            "tokens_trained": f"{params['tokens_trained'] / 1e9:.1f}B",
        },
    }

    return config


def create_training_config(model_name: str, params: dict) -> dict:
    """Create training configuration for a model."""

    config = {
        "training": {
            "model_size": model_name,
            # Training hyperparameters from paper
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,  # Overridden by max_steps
            "max_steps": params["training_steps"],
            "warmup_steps": int(params["training_steps"] * 0.01),  # 1% warmup
            # Optimizer settings (Adam with specific betas)
            "optimizer": "adamw",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            # Learning rate schedule
            "lr_scheduler_type": "cosine",
            # Mixed precision
            "fp16": True,
            "bf16": False,
            # Checkpointing
            "save_strategy": "steps",
            "save_steps": max(100, params["training_steps"] // 10),
            "save_total_limit": 5,
            "load_best_model_at_end": False,
            # Logging
            "logging_steps": 10,
            "logging_first_step": True,
            "report_to": [],  # Empty for testing
            # Evaluation
            "evaluation_strategy": "steps",
            "eval_steps": max(100, params["training_steps"] // 20),
            "per_device_eval_batch_size": min(params["batch_size"], 32),
            # Distributed training
            "ddp_find_unused_parameters": False,
            "gradient_checkpointing": False,  # Not needed for small models
            # Data
            "max_length": 2048,
            "num_workers": 4,
            # Random seed
            "seed": 42,
        }
    }

    return config


def main():
    """Update all configuration files."""

    # Base directory for configs
    config_dir = Path("../configs")
    model_config_dir = config_dir / "model_configs"
    training_config_dir = config_dir / "training_configs"

    print("Creating OLMo configuration files based on paper...")
    print("=" * 70)

    # Create all model configs
    for model_name, params in OLMO_CONFIGS.items():
        # Model configuration
        model_config = create_model_config(model_name, params)
        model_file = model_config_dir / f"olmo_{model_name.lower()}.yaml"

        print(f"\nCreating {model_file}")
        with open(model_file, "w") as f:
            yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

        # Training configuration
        training_config = create_training_config(model_name, params)
        training_file = training_config_dir / f"olmo_{model_name.lower()}_training.yaml"

        print(f"Creating {training_file}")
        with open(training_file, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)

    # Update the Python configuration dictionary
    print("\nUpdating olmo/models/configuration_olmo.py...")

    config_py_content = '''# src/models/configuration_olmo.py
from transformers import PretrainedConfig
from typing import Optional


class OLMoConfig(PretrainedConfig):
    """Configuration class for OLMo models."""

    model_type = "olmo"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swiglu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        use_bias: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_bias = use_bias
        self.tie_word_embeddings = tie_word_embeddings


# Predefined configurations for each model size (matching OLMo paper exactly)
OLMO_CONFIGS = {
'''

    # Add each configuration
    for model_name, params in OLMO_CONFIGS.items():
        intermediate = int(params["hidden_size"] * 4)
        config_py_content += f"""    "{model_name}": OLMoConfig(
        hidden_size={params["hidden_size"]},
        num_hidden_layers={params["num_layers"]},
        num_attention_heads={params["num_heads"]},
        intermediate_size={intermediate},
    ),
"""

    config_py_content += "}\n"

    # Write the updated configuration
    config_py_path = Path("../olmo/models/configuration_olmo.py")
    with open(config_py_path, "w") as f:
        f.write(config_py_content)

    print(f"\nUpdated {config_py_path}")

    print("\n" + "=" * 70)
    print("Configuration update complete!")
    print("\nKey changes:")
    print("- All model sizes now match paper exactly")
    print("- Learning rates follow the paper's schedule")
    print("- Batch sizes and training steps are correct")
    print("- Intermediate sizes use 4x expansion for SwiGLU")


if __name__ == "__main__":
    main()
