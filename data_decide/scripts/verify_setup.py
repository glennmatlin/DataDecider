#!/usr/bin/env python3
"""Verify the training setup before running full training."""

import json
from pathlib import Path

import torch
import yaml
from datasets import DatasetDict
from transformers import AutoTokenizer

from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
from data_decide.olmo.models.olmo_model import OLMoForCausalLM


def main():
    print("OLMo 4M Training Setup Verification")
    print("=" * 50)

    # 1. Check dataset
    print("\n1. Checking dataset...")
    dataset_path = Path("../data/processed/olmo_4m_400M_tokens")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return

    dataset = DatasetDict.load_from_disk(str(dataset_path))
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    print("✅ Dataset loaded successfully")
    print(f"   - Train samples: {len(dataset['train']):,}")
    print(f"   - Val samples: {len(dataset['validation']):,}")
    print(f"   - Total tokens: {metadata['total_tokens']:,}")
    print(f"   - Sequence length: {metadata['max_seq_length']}")

    # Check sample
    sample = dataset["train"][0]
    print(f"   - Sample shape: {len(sample['input_ids'])}")
    print(f"   - Keys: {list(sample.keys())}")

    # 2. Check model configuration
    print("\n2. Checking model configuration...")
    config = OLMO_CONFIGS["4M"]
    print("✅ Model config loaded")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Layers: {config.num_hidden_layers}")
    print(f"   - Heads: {config.num_attention_heads}")
    print(f"   - Vocab size: {config.vocab_size}")

    # 3. Initialize model
    print("\n3. Initializing model...")
    model = OLMoForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print("✅ Model initialized")
    print(f"   - Total parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # 4. Check training config
    print("\n4. Checking training configuration...")
    training_config_path = Path("../configs/training_configs/olmo_4m_training.yaml")
    if not training_config_path.exists():
        print(f"❌ Training config not found at {training_config_path}")
        return

    with open(training_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    print("✅ Training config loaded")
    print(f"   - Learning rate: {training_config['training']['learning_rate']}")
    print(f"   - Batch size: {training_config['training']['batch_size']}")
    print(f"   - Training steps: {training_config['training']['max_steps']}")
    print(f"   - Warmup steps: {training_config['training']['warmup_steps']}")

    # 5. Calculate training details
    print("\n5. Training calculations...")
    batch_size = training_config["training"]["batch_size"]
    max_steps = training_config["training"]["max_steps"]
    seq_length = metadata["max_seq_length"]

    total_tokens_training = batch_size * max_steps * seq_length
    epochs = total_tokens_training / metadata["total_tokens"]

    print(f"   - Tokens per step: {batch_size * seq_length:,}")
    print(f"   - Total tokens in training: {total_tokens_training:,}")
    print(f"   - Dataset epochs: {epochs:.2f}")

    # 6. Check tokenizer
    print("\n6. Checking tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print("✅ Tokenizer loaded")
    print(f"   - Vocab size: {len(tokenizer)}")
    print(f"   - Model expects: {config.vocab_size}")

    if len(tokenizer) != config.vocab_size:
        print("⚠️  Warning: Tokenizer vocab size mismatch!")

    # 7. Test forward pass
    print("\n7. Testing forward pass...")
    with torch.no_grad():
        input_ids = torch.tensor([sample["input_ids"][:100]])  # Small sample
        outputs = model(input_ids)
        print("✅ Forward pass successful")
        print(f"   - Output shape: {outputs.logits.shape}")

    print("\n" + "=" * 50)
    print("✅ All checks passed! Ready to train.")
    print("\nTo start training, run:")
    print("  cd data_decide")
    print("  ./scripts/train_4m_datadecide.sh")


if __name__ == "__main__":
    main()
