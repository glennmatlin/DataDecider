"""Tokenized dataset loader for DataDecider.

This module provides efficient loading of pre-tokenized datasets with:
- Memory-mapped loading for large datasets
- Dataset validation and compatibility checking
- Metadata access and verification
- Support for various storage formats
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import yaml
from datasets import Dataset, DatasetDict

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TokenizedDatasetInfo:
    """Information about a tokenized dataset."""

    name: str
    path: Path
    total_tokens: int
    total_sequences: int
    sequence_length: int
    tokenizer_name: str
    vocab_size: int
    train_samples: int
    validation_samples: Optional[int]
    description: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_metadata(cls, metadata_path: Path) -> "TokenizedDatasetInfo":
        """Load dataset info from metadata file."""
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return cls(
            name=metadata_path.parent.name,
            path=metadata_path.parent,
            total_tokens=metadata["statistics"]["total_tokens"],
            total_sequences=metadata["statistics"]["total_sequences"],
            sequence_length=metadata["tokenization_config"]["max_seq_length"],
            tokenizer_name=metadata["tokenizer_info"]["name"],
            vocab_size=metadata["tokenizer_info"]["vocab_size"],
            train_samples=metadata["dataset_info"]["train_samples"],
            validation_samples=metadata["dataset_info"].get("validation_samples"),
            description=metadata.get("description"),
            created_at=metadata.get("created_at"),
        )


class TokenizedDatasetLoader:
    """Loads pre-tokenized datasets efficiently."""

    def __init__(self, dataset_path: Union[str, Path], validate: bool = True):
        """Initialize dataset loader.

        Args:
            dataset_path: Path to tokenized dataset directory
            validate: Whether to validate dataset integrity
        """
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata
        self.metadata_path = self.dataset_path / "metadata.json"
        if not self.metadata_path.exists():
            raise ValueError(f"Dataset metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Create dataset info
        self.info = TokenizedDatasetInfo.from_metadata(self.metadata_path)

        # Validate if requested
        if validate:
            self._validate_dataset()

        # Load dataset lazily
        self._dataset = None

    def _validate_dataset(self):
        """Validate dataset integrity."""
        logger.info(f"Validating dataset: {self.dataset_path}")

        # Check required files
        dataset_dict_path = self.dataset_path / "dataset_dict.json"
        if not dataset_dict_path.exists():
            # Check for parquet format
            train_path = self.dataset_path / "train.parquet"
            if not train_path.exists():
                raise ValueError(f"Dataset files not found in {self.dataset_path}")

        # Verify checksums if available
        if "checksums" in self.metadata:
            logger.info("Verifying checksums...")
            for file_path, expected_checksum in self.metadata["checksums"].items():
                full_path = self.dataset_path / file_path
                if full_path.exists():
                    actual_checksum = self._calculate_checksum(full_path)
                    if actual_checksum != expected_checksum:
                        logger.warning(f"Checksum mismatch for {file_path}")

        logger.info("Dataset validation complete")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def load(self, split: Optional[str] = None) -> Union[Dataset, DatasetDict]:
        """Load the tokenized dataset.

        Args:
            split: Specific split to load ("train", "validation") or None for all

        Returns:
            Dataset or DatasetDict depending on split parameter
        """
        if self._dataset is None:
            logger.info(f"Loading dataset from {self.dataset_path}")

            # Check format
            if (self.dataset_path / "dataset_dict.json").exists():
                # Arrow format
                self._dataset = DatasetDict.load_from_disk(str(self.dataset_path))
            else:
                # Parquet format
                dataset_dict = {}
                for split_file in ["train.parquet", "validation.parquet"]:
                    split_path = self.dataset_path / split_file
                    if split_path.exists():
                        split_name = split_file.replace(".parquet", "")
                        dataset_dict[split_name] = Dataset.from_parquet(str(split_path))
                self._dataset = DatasetDict(dataset_dict)

            logger.info(f"Dataset loaded: {len(self._dataset)} splits")

        if split:
            if split not in self._dataset:
                raise ValueError(f"Split '{split}' not found in dataset")
            return self._dataset[split]
        else:
            return self._dataset

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader for the dataset.

        Args:
            split: Dataset split to load
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer

        Returns:
            PyTorch DataLoader
        """
        dataset = self.load(split)

        # Set format for PyTorch
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def verify_compatibility(self, model_config: Dict) -> Tuple[bool, Optional[str]]:
        """Verify dataset compatibility with model configuration.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Tuple of (is_compatible, error_message)
        """
        # Check sequence length
        if "max_position_embeddings" in model_config:
            model_seq_len = model_config["max_position_embeddings"]
            if self.info.sequence_length > model_seq_len:
                return (
                    False,
                    f"Dataset sequence length ({self.info.sequence_length}) exceeds model max ({model_seq_len})",
                )

        # Check vocab size
        if "vocab_size" in model_config:
            model_vocab_size = model_config["vocab_size"]
            if self.info.vocab_size != model_vocab_size:
                return False, f"Dataset vocab size ({self.info.vocab_size}) doesn't match model ({model_vocab_size})"

        return True, None

    def get_sample(self, n_samples: int = 5) -> Dict[str, torch.Tensor]:
        """Get a sample of data for testing.

        Args:
            n_samples: Number of samples to return

        Returns:
            Dictionary with sample data
        """
        dataset = self.load("train")
        indices = torch.randperm(len(dataset))[:n_samples].tolist()

        samples = dataset.select(indices)
        samples.set_format(type="torch")

        return {
            "input_ids": samples["input_ids"],
            "attention_mask": samples["attention_mask"],
            "labels": samples["labels"],
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TokenizedDatasetLoader(\n"
            f"  name: {self.info.name}\n"
            f"  path: {self.info.path}\n"
            f"  tokens: {self.info.total_tokens:,}\n"
            f"  sequences: {self.info.total_sequences:,}\n"
            f"  tokenizer: {self.info.tokenizer_name}\n"
            f")"
        )


class DatasetRegistry:
    """Registry for managing multiple tokenized datasets."""

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        """Initialize dataset registry.

        Args:
            registry_path: Path to registry YAML file
        """
        self.registry_path = Path(registry_path) if registry_path else None
        self.datasets = {}

        if self.registry_path and self.registry_path.exists():
            self.load_registry()

    def load_registry(self):
        """Load datasets from registry file."""
        with open(self.registry_path, "r") as f:
            registry_data = yaml.safe_load(f)

        for name, config in registry_data.get("datasets", {}).items():
            self.register(name, config["path"], config.get("description"))

    def register(self, name: str, path: Union[str, Path], description: Optional[str] = None):
        """Register a dataset.

        Args:
            name: Dataset name
            path: Path to dataset
            description: Optional description
        """
        dataset_path = Path(path)
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return

        try:
            loader = TokenizedDatasetLoader(dataset_path, validate=False)
            loader.info.description = description or loader.info.description
            self.datasets[name] = loader
            logger.info(f"Registered dataset: {name}")
        except Exception as e:
            logger.error(f"Failed to register dataset {name}: {e}")

    def get(self, name: str) -> TokenizedDatasetLoader:
        """Get a dataset by name.

        Args:
            name: Dataset name

        Returns:
            TokenizedDatasetLoader instance
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found in registry")
        return self.datasets[name]

    def list_datasets(self) -> Dict[str, TokenizedDatasetInfo]:
        """List all registered datasets.

        Returns:
            Dictionary of dataset names to info
        """
        return {name: loader.info for name, loader in self.datasets.items()}

    def save_registry(self, output_path: Optional[Union[str, Path]] = None):
        """Save registry to YAML file.

        Args:
            output_path: Output path (uses self.registry_path if not provided)
        """
        output_path = Path(output_path) if output_path else self.registry_path
        if not output_path:
            raise ValueError("No output path provided")

        registry_data = {
            "datasets": {
                name: {
                    "path": str(loader.info.path),
                    "tokens": loader.info.total_tokens,
                    "description": loader.info.description,
                }
                for name, loader in self.datasets.items()
            }
        }

        with open(output_path, "w") as f:
            yaml.dump(registry_data, f, default_flow_style=False)

        logger.info(f"Registry saved to {output_path}")


# Convenience functions
def load_dataset(path: Union[str, Path], **kwargs) -> Union[Dataset, DatasetDict]:
    """Load a tokenized dataset.

    Args:
        path: Path to dataset
        **kwargs: Additional arguments for TokenizedDatasetLoader

    Returns:
        Loaded dataset
    """
    loader = TokenizedDatasetLoader(path, **kwargs)
    return loader.load()


def create_registry(base_path: Union[str, Path] = "data/tokenized") -> DatasetRegistry:
    """Create a dataset registry by scanning a directory.

    Args:
        base_path: Base path to scan for datasets

    Returns:
        DatasetRegistry instance
    """
    registry = DatasetRegistry()
    base_path = Path(base_path)

    # Scan for datasets
    for dataset_path in base_path.rglob("metadata.json"):
        dataset_dir = dataset_path.parent
        dataset_name = dataset_dir.relative_to(base_path).as_posix().replace("/", "_")
        registry.register(dataset_name, dataset_dir)

    return registry
