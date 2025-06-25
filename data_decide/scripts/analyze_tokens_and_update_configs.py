#!/usr/bin/env python3
"""Analyze token requirements and update configurations based on OLMo paper."""

import os
import sys
import gzip
import json
import yaml
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# OLMo hyperparameters from the paper
OLMO_HYPERPARAMS = {
    "4M": {
        "actual_size": "3.7M",
        "batch_size": 32,
        "hidden_dim": 64,
        "learning_rate": 1.4e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 5725,
        "tokens_trained": 0.4e9,  # 0.4B
    },
    "6M": {
        "actual_size": "6.0M",
        "batch_size": 32,
        "hidden_dim": 96,
        "learning_rate": 1.2e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 9182,
        "tokens_trained": 0.6e9,  # 0.6B
    },
    "8M": {
        "actual_size": "8.5M",
        "batch_size": 32,
        "hidden_dim": 128,
        "learning_rate": 1.1e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 13039,
        "tokens_trained": 0.9e9,  # 0.9B
    },
    "10M": {
        "actual_size": "9.9M",
        "batch_size": 32,
        "hidden_dim": 144,
        "learning_rate": 1.0e-02,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 15117,
        "tokens_trained": 1.0e9,  # 1.0B
    },
    "14M": {
        "actual_size": "14.4M",
        "batch_size": 32,
        "hidden_dim": 192,
        "learning_rate": 9.2e-03,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 21953,
        "tokens_trained": 1.4e9,  # 1.4B
    },
    "16M": {
        "actual_size": "16.0M",
        "batch_size": 32,
        "hidden_dim": 208,
        "learning_rate": 8.9e-03,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 24432,
        "tokens_trained": 1.6e9,  # 1.6B
    },
    "20M": {
        "actual_size": "19.1M",
        "batch_size": 64,
        "hidden_dim": 192,
        "learning_rate": 8.4e-03,
        "num_heads": 8,
        "num_layers": 8,
        "training_steps": 14584,
        "tokens_trained": 1.9e9,  # 1.9B
    },
    "60M": {
        "actual_size": "57.1M",
        "batch_size": 96,
        "hidden_dim": 384,
        "learning_rate": 5.8e-03,
        "num_heads": 12,
        "num_layers": 16,
        "training_steps": 29042,
        "tokens_trained": 5.7e9,  # 5.7B
    },
    "90M": {
        "actual_size": "97.9M",
        "batch_size": 160,
        "hidden_dim": 528,
        "learning_rate": 4.9e-03,
        "num_heads": 12,
        "num_layers": 16,
        "training_steps": 29901,
        "tokens_trained": 9.8e9,  # 9.8B
    },
    "150M": {
        "actual_size": "151.9M",
        "batch_size": 192,
        "hidden_dim": 768,
        "learning_rate": 4.2e-03,
        "num_heads": 12,
        "num_layers": 12,
        "training_steps": 38157,
        "tokens_trained": 15.0e9,  # 15.0B
    },
    "300M": {
        "actual_size": "320.0M",
        "batch_size": 320,
        "hidden_dim": 1024,
        "learning_rate": 3.3e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 45787,
        "tokens_trained": 30.0e9,  # 30.0B
    },
    "530M": {
        "actual_size": "530.1M",
        "batch_size": 448,
        "hidden_dim": 1344,
        "learning_rate": 2.8e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 57786,
        "tokens_trained": 53.0e9,  # 53.0B
    },
    "750M": {
        "actual_size": "681.3M",  # Note: actual is 681.3M, not 750M
        "batch_size": 576,
        "hidden_dim": 1536,
        "learning_rate": 2.5e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 63589,
        "tokens_trained": 75.0e9,  # 75.0B
    },
    "1B": {
        "actual_size": "1176.8M",
        "batch_size": 704,
        "hidden_dim": 2048,
        "learning_rate": 2.1e-03,
        "num_heads": 16,
        "num_layers": 16,
        "training_steps": 69369,
        "tokens_trained": 100.0e9,  # 100.0B
    },
}

def count_tokens_sample(file_path: str, tokenizer, sample_size: int = 1000):
    """Sample documents to estimate total tokens."""
    
    token_counts = []
    
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            doc = json.loads(line)
            tokens = tokenizer.encode(doc['text'], truncation=False)
            token_counts.append(len(tokens))
    
    return np.array(token_counts)

def analyze_token_requirements(arxiv_file: str, sequence_length: int = 2048):
    """Analyze token requirements for all models."""
    
    # Load tokenizer - GPT-NeoX-20B as used in OLMo
    print("Loading GPT-NeoX-20B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Sample to estimate tokens
    print(f"\nSampling documents from {arxiv_file}...")
    token_counts = count_tokens_sample(arxiv_file, tokenizer, sample_size=1000)
    avg_tokens_per_doc = np.mean(token_counts)
    
    # Total documents from previous count
    total_docs = 12966
    estimated_total_tokens = int(avg_tokens_per_doc * total_docs)
    
    print(f"\nToken Analysis:")
    print(f"Average tokens per document: {avg_tokens_per_doc:.1f}")
    print(f"Total documents: {total_docs:,}")
    print(f"Estimated total tokens available: {estimated_total_tokens:,}")
    
    print("\n" + "="*80)
    print("Model Requirements vs Available Data")
    print("="*80)
    
    for model_name, params in OLMO_HYPERPARAMS.items():
        tokens_required = int(params['tokens_trained'])
        batch_size = params['batch_size']
        training_steps = params['training_steps']
        
        # Calculate tokens per step
        tokens_per_step = batch_size * sequence_length
        
        # Verify the calculation
        calculated_tokens = tokens_per_step * training_steps
        
        print(f"\n{model_name} Model ({params['actual_size']} params):")
        print(f"  Required tokens: {tokens_required:,}")
        print(f"  Training steps: {training_steps:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Tokens per step: {tokens_per_step:,} (batch_size × seq_length)")
        print(f"  Calculated total: {calculated_tokens:,}")
        
        if model_name == "4M":
            if estimated_total_tokens >= tokens_required:
                print(f"  Data status: ✅ SUFFICIENT ({estimated_total_tokens / tokens_required:.2f}x required)")
            else:
                epochs_needed = tokens_required / estimated_total_tokens
                print(f"  Data status: ❌ INSUFFICIENT")
                print(f"  Need to repeat data {epochs_needed:.1f} times")
                
            # Calculate actual training parameters
            print(f"\n  Training Configuration:")
            print(f"    Learning rate: {params['learning_rate']}")
            print(f"    Hidden dimension: {params['hidden_dim']}")
            print(f"    Number of heads: {params['num_heads']}")
            print(f"    Number of layers: {params['num_layers']}")
            print(f"    Intermediate size: {params['hidden_dim'] * 4}")  # Standard 4x expansion
    
    return estimated_total_tokens

def generate_updated_configs():
    """Generate configuration files for all models based on paper."""
    
    configs_dir = Path("../configs")
    
    # Create model configs
    for model_name, params in OLMO_HYPERPARAMS.items():
        model_config = {
            "model": {
                "name": f"olmo-{model_name.lower()}",
                "vocab_size": 50257,  # GPT-NeoX tokenizer size
                "hidden_size": params["hidden_dim"],
                "num_hidden_layers": params["num_layers"],
                "num_attention_heads": params["num_heads"],
                "intermediate_size": params["hidden_dim"] * 4,  # Standard 4x
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
                "batch_size": params["batch_size"],
                "learning_rate": params["learning_rate"],
                "training_steps": params["training_steps"],
                "tokens_trained": params["tokens_trained"] / 1e9,  # In billions
                "actual_model_size": params["actual_size"],
            }
        }
        
        config_path = configs_dir / "model_configs" / f"olmo_{model_name.lower()}.yaml"
        print(f"\nWould create: {config_path}")
        print(yaml.dump(model_config, default_flow_style=False)[:200] + "...")
        
    # Create training configs
    for model_name, params in OLMO_HYPERPARAMS.items():
        if model_name == "4M":  # Detailed config for 4M
            training_config = {
                "training": {
                    "model_size": model_name,
                    "learning_rate": params["learning_rate"],
                    "batch_size": params["batch_size"],
                    "gradient_accumulation_steps": 1,
                    "num_train_epochs": 1,  # Will be overridden by max_steps
                    "max_steps": params["training_steps"],
                    "warmup_steps": int(params["training_steps"] * 0.01),  # 1% warmup
                    "weight_decay": 0.1,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "adam_epsilon": 1e-8,
                    "max_grad_norm": 1.0,
                    "optimizer": "adamw",
                    "lr_scheduler_type": "cosine",
                    "fp16": True,
                    "bf16": False,
                    "save_strategy": "steps",
                    "save_steps": 1000,
                    "save_total_limit": 5,
                    "logging_steps": 10,
                    "eval_steps": 500,
                    "per_device_eval_batch_size": 32,
                    "max_length": 2048,
                    "num_workers": 4,
                    "seed": 42,
                }
            }
            
            config_path = configs_dir / "training_configs" / f"olmo_{model_name.lower()}_training.yaml"
            print(f"\nWould create: {config_path}")
            print(yaml.dump(training_config, default_flow_style=False))

def main():
    # Find arxiv data
    arxiv_file = None
    for path in ["../../tests/test_data/arxiv-0098.json.gz", "../tests/test_data/arxiv-0098.json.gz"]:
        if os.path.exists(path):
            arxiv_file = path
            break
    
    if not arxiv_file:
        print("Error: Could not find arxiv-0098.json.gz")
        return
    
    # Analyze token requirements
    total_tokens = analyze_token_requirements(arxiv_file)
    
    # Show config generation
    print("\n" + "="*80)
    print("Configuration Updates Needed")
    print("="*80)
    generate_updated_configs()
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\nFor 4M model training:")
    print("1. We need 400M tokens total")
    print("2. With 32 batch size and 2048 sequence length = 65,536 tokens per step")
    print("3. 5,725 steps × 65,536 tokens/step ≈ 375M tokens")
    print("4. If we have less data, we need to repeat the dataset multiple times")
    print("\nNext steps:")
    print("- Count exact tokens in arxiv-0098.json.gz")
    print("- Update all configuration files with correct hyperparameters")
    print("- Implement data repetition if needed")

if __name__ == "__main__":
    main()