# src/data/preprocessing.py
import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer
from datasets import Dataset


class OLMoDataPreprocessor:
    """Preprocessing pipeline for OLMo training data."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        concatenate_documents: bool = True,
        add_eos_token: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concatenate_documents = concatenate_documents
        self.add_eos_token = add_eos_token

        # Add special tokens if not present
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    def preprocess_batch(self, examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of examples."""
        texts = examples["text"]

        # Add EOS tokens if specified
        if self.add_eos_token:
            texts = [text + self.tokenizer.eos_token for text in texts]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )

        if self.concatenate_documents:
            # Concatenate all documents
            concatenated = self._concatenate_documents(tokenized)
            return concatenated
        else:
            return tokenized

    def _concatenate_documents(self, tokenized: Dict[str, List]) -> Dict[str, List]:
        """Concatenate documents and chunk into max_length sequences."""
        concatenated_ids = []
        concatenated_attention_mask = []

        # Flatten all input_ids
        all_ids = []
        for ids in tokenized["input_ids"]:
            all_ids.extend(ids)

        # Chunk into sequences of max_length
        for i in range(0, len(all_ids), self.max_length):
            chunk = all_ids[i : i + self.max_length]
            if len(chunk) == self.max_length:
                concatenated_ids.append(chunk)
                concatenated_attention_mask.append([1] * self.max_length)

        return {
            "input_ids": concatenated_ids,
            "attention_mask": concatenated_attention_mask,
            "labels": concatenated_ids.copy(),  # For causal LM
        }

    def create_training_dataset(
        self, raw_dataset: Dataset, num_proc: int = 4, batch_size: int = 1000
    ) -> Dataset:
        """Create training dataset with preprocessing."""
        # Remove unnecessary columns
        columns_to_remove = [
            col for col in raw_dataset.column_names if col not in ["text"]
        ]

        # Apply preprocessing
        processed_dataset = raw_dataset.map(
            self.preprocess_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=columns_to_remove,
        )

        # Set format for PyTorch
        processed_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        return processed_dataset
