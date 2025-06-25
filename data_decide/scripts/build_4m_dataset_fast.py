#!/usr/bin/env python3
"""Fast builder for 400M token dataset - processes in batches."""

import os
import gzip
import json
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import time
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Fast4MDatasetBuilder:
    """Fast dataset builder with batch processing and checkpointing."""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.max_seq_length = 2048
        self.target_tokens = 400_000_000
        self.batch_size = 100  # Process 100 docs at a time
        self.output_dir = Path("../data/processed/olmo_4m_400M_tokens")
        self.checkpoint_dir = Path("../data/processed/checkpoints")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Target: {self.target_tokens:,} tokens")
        logger.info(f"Batch size: {self.batch_size} documents")
    
    def load_checkpoint(self):
        """Load checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / "4m_dataset_checkpoint.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint['total_tokens']:,} tokens processed")
            return checkpoint
        return None
    
    def save_checkpoint(self, all_token_ids, total_tokens, file_idx, doc_idx):
        """Save checkpoint."""
        checkpoint = {
            'all_token_ids': all_token_ids,
            'total_tokens': total_tokens,
            'file_idx': file_idx,
            'doc_idx': doc_idx,
        }
        checkpoint_file = self.checkpoint_dir / "4m_dataset_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def build_dataset(self):
        """Build dataset with batch processing."""
        start_time = time.time()
        
        # Check for checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint:
            all_token_ids = checkpoint['all_token_ids']
            total_tokens = checkpoint['total_tokens']
            start_file_idx = checkpoint['file_idx']
            start_doc_idx = checkpoint['doc_idx']
        else:
            all_token_ids = []
            total_tokens = 0
            start_file_idx = 0
            start_doc_idx = 0
        
        # Files to process
        files = [
            "../data/raw/arxiv-0098.json.gz",
            "../data/raw/arxiv-0012.json.gz",
        ]
        
        current_chunk = []
        
        # Process files
        for file_idx, file_path in enumerate(files[start_file_idx:], start_file_idx):
            if total_tokens >= self.target_tokens:
                break
            
            logger.info(f"\nProcessing {Path(file_path).name}...")
            
            # Batch processing
            batch_texts = []
            doc_count = 0
            
            with gzip.open(file_path, 'rt') as f:
                # Skip already processed docs
                for _ in range(start_doc_idx):
                    next(f, None)
                start_doc_idx = 0  # Reset for next file
                
                pbar = tqdm(desc="Reading documents", unit="docs")
                
                for line in f:
                    if total_tokens >= self.target_tokens:
                        break
                    
                    doc = json.loads(line)
                    batch_texts.append(doc['text'])
                    doc_count += 1
                    pbar.update(1)
                    
                    # Process batch
                    if len(batch_texts) >= self.batch_size:
                        # Tokenize batch
                        batch_tokens = self.tokenizer(
                            batch_texts, 
                            truncation=False, 
                            add_special_tokens=False
                        )['input_ids']
                        
                        # Process each document's tokens
                        for tokens in batch_tokens:
                            tokens.append(self.tokenizer.eos_token_id)
                            current_chunk.extend(tokens)
                            
                            # Extract sequences
                            while len(current_chunk) >= self.max_seq_length:
                                all_token_ids.append(current_chunk[:self.max_seq_length])
                                current_chunk = current_chunk[self.max_seq_length:]
                                total_tokens += self.max_seq_length
                        
                        pbar.set_postfix({
                            'tokens': f"{total_tokens:,}",
                            'sequences': len(all_token_ids),
                            'progress': f"{total_tokens/self.target_tokens*100:.1f}%"
                        })
                        
                        # Clear batch
                        batch_texts = []
                        
                        # Save checkpoint every 10k sequences
                        if len(all_token_ids) % 10000 == 0:
                            self.save_checkpoint(all_token_ids, total_tokens, file_idx, doc_count)
                
                # Process remaining batch
                if batch_texts:
                    batch_tokens = self.tokenizer(
                        batch_texts, 
                        truncation=False, 
                        add_special_tokens=False
                    )['input_ids']
                    
                    for tokens in batch_tokens:
                        tokens.append(self.tokenizer.eos_token_id)
                        current_chunk.extend(tokens)
                        
                        while len(current_chunk) >= self.max_seq_length:
                            all_token_ids.append(current_chunk[:self.max_seq_length])
                            current_chunk = current_chunk[self.max_seq_length:]
                            total_tokens += self.max_seq_length
                
                pbar.close()
            
            logger.info(f"Processed {doc_count} documents, total tokens: {total_tokens:,}")
        
        # Trim to exact target
        if total_tokens > self.target_tokens:
            sequences_needed = self.target_tokens // self.max_seq_length
            all_token_ids = all_token_ids[:sequences_needed]
            total_tokens = sequences_needed * self.max_seq_length
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\nCreating dataset splits...")
        
        # Create splits
        n_sequences = len(all_token_ids)
        n_val = int(n_sequences * 0.05)
        n_train = n_sequences - n_val
        
        # Create HuggingFace dataset
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict({
                'input_ids': all_token_ids[:n_train],
                'attention_mask': [[1] * self.max_seq_length] * n_train,
                'labels': all_token_ids[:n_train],
            }),
            'validation': Dataset.from_dict({
                'input_ids': all_token_ids[n_train:],
                'attention_mask': [[1] * self.max_seq_length] * n_val,
                'labels': all_token_ids[n_train:],
            })
        })
        
        # Save dataset
        logger.info(f"Saving dataset to {self.output_dir}...")
        dataset_dict.save_to_disk(str(self.output_dir))
        
        # Save metadata
        metadata = {
            'tokenizer': self.tokenizer.name_or_path,
            'vocab_size': self.tokenizer.vocab_size,
            'max_seq_length': self.max_seq_length,
            'total_sequences': n_sequences,
            'train_sequences': n_train,
            'validation_sequences': n_val,
            'total_tokens': total_tokens,
            'processing_time_seconds': elapsed_time,
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up checkpoint
        checkpoint_file = self.checkpoint_dir / "4m_dataset_checkpoint.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        print("\n" + "="*70)
        print("DATASET CREATION COMPLETE")
        print("="*70)
        print(f"Total sequences: {n_sequences:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Time: {elapsed_time:.1f} seconds")
        print(f"Output: {self.output_dir}")
        print("="*70)
        
        return dataset_dict


def main():
    builder = Fast4MDatasetBuilder()
    dataset = builder.build_dataset()
    
    # Quick verification
    print("\nVerifying dataset...")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['validation'])}")
    print("✅ Dataset ready!")


if __name__ == "__main__":
    main()