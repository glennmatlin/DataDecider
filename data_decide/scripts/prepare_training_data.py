#!/usr/bin/env python3
"""Prepare arXiv data for OLMo 4M model training.

This script:
1. Loads all arXiv JSON.gz files
2. Tokenizes text using GPT-NeoX-20B tokenizer
3. Concatenates documents and chunks into sequences
4. Saves preprocessed data for training
"""

import os
import sys
import gzip
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data preparation."""
    raw_data_dir: str = "../data/raw"
    processed_data_dir: str = "../data/processed"
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    validation_split: float = 0.05
    target_tokens: int = 400_000_000  # 0.4B tokens for 4M model
    save_format: str = "parquet"  # or "arrow"
    batch_size: int = 1000  # Process documents in batches


class ArxivDataProcessor:
    """Process arXiv data for OLMo training."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Ensure EOS token is set
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"Initialized tokenizer: {config.tokenizer_name}")
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
        logger.info(f"EOS token: {self.tokenizer.eos_token} (ID: {self.eos_token_id})")
    
    def load_documents(self) -> Iterator[str]:
        """Load all documents from arXiv JSON.gz files."""
        data_files = sorted(Path(self.config.raw_data_dir).glob("arxiv-*.json.gz"))
        logger.info(f"Found {len(data_files)} arXiv files")
        
        for file_path in data_files:
            logger.info(f"Loading {file_path.name}")
            with gzip.open(file_path, 'rt') as f:
                for line in f:
                    doc = json.loads(line)
                    yield doc['text']
    
    def count_total_tokens(self) -> Tuple[int, int]:
        """Count total tokens and documents in the dataset."""
        logger.info("Counting total tokens...")
        total_tokens = 0
        total_docs = 0
        
        for doc in tqdm(self.load_documents(), desc="Counting tokens"):
            tokens = self.tokenizer.encode(doc, truncation=False)
            total_tokens += len(tokens)
            total_docs += 1
        
        logger.info(f"Total documents: {total_docs:,}")
        logger.info(f"Total tokens: {total_tokens:,}")
        return total_tokens, total_docs
    
    def tokenize_and_concatenate(self) -> List[List[int]]:
        """Tokenize documents and concatenate into fixed-length sequences."""
        logger.info("Tokenizing and concatenating documents...")
        
        all_token_ids = []
        current_chunk = []
        
        for doc in tqdm(self.load_documents(), desc="Tokenizing"):
            # Tokenize document
            tokens = self.tokenizer.encode(doc, truncation=False)
            
            # Add EOS token
            tokens.append(self.eos_token_id)
            
            # Add to current chunk
            current_chunk.extend(tokens)
            
            # Extract complete sequences
            while len(current_chunk) >= self.config.max_seq_length:
                all_token_ids.append(current_chunk[:self.config.max_seq_length])
                current_chunk = current_chunk[self.config.max_seq_length:]
        
        # Handle remaining tokens
        if current_chunk:
            # Pad the last chunk
            padding_length = self.config.max_seq_length - len(current_chunk)
            current_chunk.extend([self.tokenizer.pad_token_id or self.eos_token_id] * padding_length)
            all_token_ids.append(current_chunk)
        
        logger.info(f"Created {len(all_token_ids):,} sequences of length {self.config.max_seq_length}")
        return all_token_ids
    
    def create_dataset(self, token_ids: List[List[int]]) -> DatasetDict:
        """Create HuggingFace dataset from token IDs."""
        logger.info("Creating dataset...")
        
        # Calculate split
        n_sequences = len(token_ids)
        n_validation = int(n_sequences * self.config.validation_split)
        n_train = n_sequences - n_validation
        
        # Shuffle sequences
        indices = np.random.permutation(n_sequences)
        
        # Split data
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_data = {
            'input_ids': [token_ids[i] for i in train_indices],
            'labels': [token_ids[i] for i in train_indices],  # For causal LM, labels = input_ids
        }
        
        val_data = {
            'input_ids': [token_ids[i] for i in val_indices],
            'labels': [token_ids[i] for i in val_indices],
        }
        
        # Create attention masks (all 1s since we're using full sequences)
        train_data['attention_mask'] = [[1] * self.config.max_seq_length for _ in range(n_train)]
        val_data['attention_mask'] = [[1] * self.config.max_seq_length for _ in range(n_validation)]
        
        # Create datasets
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        logger.info(f"Train sequences: {len(train_dataset):,}")
        logger.info(f"Validation sequences: {len(val_dataset):,}")
        logger.info(f"Total tokens (train): {len(train_dataset) * self.config.max_seq_length:,}")
        
        return dataset_dict
    
    def check_token_requirements(self, total_tokens: int) -> float:
        """Check if we have enough tokens and calculate epochs needed."""
        epochs_needed = self.config.target_tokens / total_tokens
        
        if total_tokens >= self.config.target_tokens:
            logger.info(f"✅ Sufficient tokens: {total_tokens:,} >= {self.config.target_tokens:,}")
            logger.info(f"Can train for 1 epoch")
        else:
            logger.info(f"⚠️ Insufficient tokens: {total_tokens:,} < {self.config.target_tokens:,}")
            logger.info(f"Need {epochs_needed:.2f} epochs to reach target")
        
        return epochs_needed
    
    def save_dataset(self, dataset: DatasetDict):
        """Save dataset to disk."""
        output_dir = Path(self.config.processed_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        if self.config.save_format == "parquet":
            dataset.save_to_disk(output_dir / "olmo_4m_dataset", num_proc=4)
            logger.info(f"Saved dataset to {output_dir / 'olmo_4m_dataset'}")
        else:
            dataset.save_to_disk(output_dir / "olmo_4m_dataset")
            logger.info(f"Saved dataset to {output_dir / 'olmo_4m_dataset'}")
        
        # Save metadata
        metadata = {
            'tokenizer': self.config.tokenizer_name,
            'max_seq_length': self.config.max_seq_length,
            'total_train_tokens': len(dataset['train']) * self.config.max_seq_length,
            'total_val_tokens': len(dataset['validation']) * self.config.max_seq_length,
            'train_sequences': len(dataset['train']),
            'val_sequences': len(dataset['validation']),
            'target_tokens': self.config.target_tokens,
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Saved dataset metadata")
    
    def process(self):
        """Main processing pipeline."""
        logger.info("Starting data preparation for OLMo 4M model")
        
        # First pass: count tokens
        total_tokens, total_docs = self.count_total_tokens()
        epochs_needed = self.check_token_requirements(total_tokens)
        
        # Second pass: tokenize and create dataset
        token_ids = self.tokenize_and_concatenate()
        
        # Handle data repetition if needed
        if epochs_needed > 1:
            logger.info(f"Repeating data {int(np.ceil(epochs_needed))} times to reach target tokens")
            original_sequences = len(token_ids)
            
            # Calculate how many times to repeat
            repeat_times = int(np.ceil(epochs_needed))
            
            # Repeat sequences
            repeated_ids = []
            for _ in range(repeat_times):
                repeated_ids.extend(token_ids)
            
            # Trim to exact target
            target_sequences = self.config.target_tokens // self.config.max_seq_length
            token_ids = repeated_ids[:target_sequences]
            
            logger.info(f"Repeated {original_sequences} sequences → {len(token_ids)} sequences")
        
        # Create dataset
        dataset = self.create_dataset(token_ids)
        
        # Save dataset
        self.save_dataset(dataset)
        
        logger.info("✅ Data preparation complete!")
        
        # Print summary
        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)
        print(f"Total documents processed: {total_docs:,}")
        print(f"Total tokens in raw data: {total_tokens:,}")
        print(f"Target tokens for 4M model: {self.config.target_tokens:,}")
        print(f"Epochs needed: {epochs_needed:.2f}")
        print(f"Total sequences created: {len(token_ids):,}")
        print(f"Sequence length: {self.config.max_seq_length}")
        print(f"Train sequences: {len(dataset['train']):,}")
        print(f"Validation sequences: {len(dataset['validation']):,}")
        print(f"Output directory: {self.config.processed_data_dir}")
        print("="*60)


def main():
    """Main entry point."""
    config = DataConfig()
    
    # Check if raw data exists
    raw_dir = Path(config.raw_data_dir)
    if not raw_dir.exists() or not list(raw_dir.glob("arxiv-*.json.gz")):
        logger.error(f"No arXiv files found in {raw_dir}")
        sys.exit(1)
    
    # Process data
    processor = ArxivDataProcessor(config)
    processor.process()


if __name__ == "__main__":
    main()