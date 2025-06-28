"""
Shared pytest fixtures for tokenization tests.
"""

import gzip
import json
import shutil

# Import directly from scripts
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, "/home/gmatlin/Codespace/DataDecider/data_decide/scripts")
from unified_tokenizer import TokenizationConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_jsonl_data():
    """Sample JSONL data for testing."""
    return [
        {"text": "The quick brown fox jumps over the lazy dog. " * 10},  # Repeat to ensure enough tokens
        {"text": "Machine learning models require large amounts of training data. " * 10},
        {"text": "Natural language processing has advanced significantly in recent years. " * 10},
        {"text": "Deep neural networks can learn complex patterns from data. " * 10},
        {"text": "Transformers have revolutionized the field of NLP. " * 10},
    ]


@pytest.fixture
def create_jsonl_file(temp_dir):
    """Factory fixture to create JSONL test files."""

    def _create_file(filename: str, data: List[Dict[str, Any]], compress: bool = False):
        file_path = temp_dir / filename

        if compress:
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

        return file_path

    return _create_file


@pytest.fixture
def create_text_file(temp_dir):
    """Factory fixture to create text test files."""

    def _create_file(filename: str, content: str):
        file_path = temp_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    return _create_file


@pytest.fixture
def base_config(temp_dir):
    """Base tokenization configuration for testing."""
    return {
        "input_path": str(temp_dir),
        "output_path": str(temp_dir / "output"),
        "tokenizer_name": "EleutherAI/gpt-neox-20b",
        "max_seq_length": 512,
        "batch_size": 10,
        "chunk_size": 100,
        "sequences_per_save": 50,
        "enable_rich_ui": False,  # Disable UI for tests
        "enable_checkpoint": True,
        "checkpoint_dir": str(temp_dir / "checkpoints"),
    }


@pytest.fixture
def mock_tokenizer_config(base_config):
    """Create a TokenizationConfig instance for testing."""
    return TokenizationConfig(**base_config)


@pytest.fixture
def large_dataset(create_jsonl_file):
    """Create a large dataset for performance testing."""
    data = []
    for i in range(1000):
        text = f"Document {i}: " + " ".join([f"word{j}" for j in range(50)])
        data.append({"text": text})

    return create_jsonl_file("large_dataset.jsonl", data)


@pytest.fixture
def corrupt_dataset(create_jsonl_file, temp_dir):
    """Create a dataset with corrupt/edge case data."""
    file_path = temp_dir / "corrupt_data.jsonl"

    with open(file_path, "w") as f:
        # Valid document
        f.write(json.dumps({"text": "Valid document 1"}) + "\n")
        # Corrupt JSON
        f.write("{invalid json syntax}\n")
        # Missing text field
        f.write(json.dumps({"content": "Missing text field"}) + "\n")
        # Empty text
        f.write(json.dumps({"text": ""}) + "\n")
        # Valid document
        f.write(json.dumps({"text": "Valid document 2"}) + "\n")
        # Very long document
        f.write(json.dumps({"text": " ".join([f"word{i}" for i in range(10000)])}) + "\n")

    return file_path


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking."""

    class Benchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

        def assert_faster_than(self, seconds: float, message: str = ""):
            assert self.elapsed is not None, "Benchmark not completed"
            assert self.elapsed < seconds, f"Too slow: {self.elapsed:.2f}s > {seconds}s. {message}"

    return Benchmark()


@pytest.fixture
def mock_rich_console(monkeypatch):
    """Mock Rich console to prevent output during tests."""

    class MockConsole:
        def print(self, *args, **kwargs):
            pass

    monkeypatch.setattr("unified_tokenizer.console", MockConsole())


@pytest.fixture(autouse=True)
def cleanup_checkpoints(temp_dir):
    """Ensure checkpoint directories are cleaned up after tests."""
    yield
    # Additional cleanup if needed
    checkpoint_dirs = list(Path.cwd().glob("tokenization_checkpoints*"))
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
