#!/usr/bin/env python3
"""Test script for 4M parameter OLMo model with ArXiv data."""

import os
import sys
import argparse
import json
import gzip
from pathlib import Path
import torch
from datetime import datetime

from data_decide.olmo.data.data_curation import DataDecideCurator
from data_decide.olmo.data.preprocessing import OLMoDataPreprocessor
from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
from data_decide.olmo.models.olmo_model import OLMoForCausalLM
from data_decide.olmo.training.trainer import OLMoTrainer
from data_decide.olmo.utils.logging_utils import setup_logging, get_logger
from data_decide.olmo.utils.config_utils import load_config, merge_configs
from transformers import AutoTokenizer
from datasets import Dataset

logger = get_logger(__name__)


def load_arxiv_sample(data_path: str, num_samples: int = 100):
    """Load a sample of ArXiv documents."""
    logger.info(f"Loading {num_samples} ArXiv documents from {data_path}")
    
    documents = []
    arxiv_file = Path(data_path) / "arxiv_sample.json.gz"
    
    if not arxiv_file.exists():
        # Try the original file
        arxiv_file = Path(data_path).parent.parent / "tests/test_data/arxiv-0098.json.gz"
    
    if not arxiv_file.exists():
        raise FileNotFoundError(f"ArXiv data not found. Looked in {arxiv_file}")
    
    with gzip.open(arxiv_file, 'rt') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            doc = json.loads(line)
            documents.append({
                'text': doc['text'][:5000],  # Limit text length for testing
                'url': doc['url']
            })
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def test_4m_model():
    """Test the 4M parameter model with minimal data."""
    # Setup
    output_dir = Path("./test_outputs/4m_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(output_dir))
    logger.info("Starting 4M model test")
    
    # Load configurations
    training_config = load_config("configs/training_configs/test_4m_training.yaml")
    model_config = OLMO_CONFIGS["4M"]
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    logger.info("Loading ArXiv test data")
    test_data_path = Path(__file__).parent.parent / "tests/test_data"
    documents = load_arxiv_sample(str(test_data_path), num_samples=50)
    
    # Create dataset
    logger.info("Creating dataset")
    raw_dataset = Dataset.from_list(documents)
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = OLMoDataPreprocessor(
        tokenizer=tokenizer,
        max_length=training_config['training']['max_length'],
        concatenate_documents=True,
        add_eos_token=True
    )
    
    train_dataset = preprocessor.create_training_dataset(
        raw_dataset,
        num_proc=1,
        batch_size=100
    )
    
    # Split for eval
    logger.info("Splitting dataset")
    if len(train_dataset) > 10:
        split = train_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        eval_dataset = None
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Initialize model
    logger.info("Initializing 4M parameter model")
    model = OLMoForCausalLM(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params / 1e6:.1f}M parameters")
    
    # Test forward pass
    logger.info("Testing forward pass")
    sample_batch = train_dataset[:1]
    with torch.no_grad():
        outputs = model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels']
        )
    logger.info(f"Forward pass successful. Loss: {outputs.loss.item():.4f}")
    
    # Initialize trainer
    logger.info("Initializing trainer")
    training_config['output_dir'] = str(output_dir)
    
    trainer = OLMoTrainer(
        config=training_config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    # Run training
    logger.info("Starting training")
    initial_loss = outputs.loss.item()
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Final evaluation
        if eval_dataset:
            logger.info("Running final evaluation")
            eval_metrics = trainer.evaluate()
            logger.info(f"Final eval metrics: {eval_metrics}")
            
            # Check if loss decreased
            if 'loss' in eval_metrics and eval_metrics['loss'] < initial_loss:
                logger.info("✓ Model is learning (loss decreased)")
            else:
                logger.warning("! Loss did not decrease as expected")
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        logger.info(f"Model saved to {final_model_path}")
        
        # Test loading checkpoint
        logger.info("Testing checkpoint loading")
        loaded_model = OLMoForCausalLM.from_pretrained(str(final_model_path))
        logger.info("✓ Checkpoint loaded successfully")
        
        # Test inference
        logger.info("Testing inference")
        test_text = "The quantum mechanical"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = loaded_model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        print("\n" + "="*50)
        print("✅ All tests passed!")
        print("="*50)
        print(f"Model parameters: {total_params:,}")
        print(f"Training steps: {training_config['training']['max_steps']}")
        print(f"Final output directory: {output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 4M OLMo model")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    test_4m_model()