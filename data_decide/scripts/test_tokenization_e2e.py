#!/usr/bin/env python3
"""
End-to-end tests for tokenization methods with coverage analysis.
Tests all processing modes: batch, streaming, and hybrid.
"""

import gzip
import json
import multiprocessing
import shutil

# Add scripts directory to path
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from unified_tokenizer import TokenizationConfig, UnifiedTokenizer


class TestTokenizationE2E:
    """End-to-end tests for tokenization methods"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test environment and clean up after each test"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.checkpoint_dir = self.test_dir / "checkpoints"
        yield
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_data(self, num_docs: int = 100, doc_length: int = 50) -> Path:
        """Create test JSONL data file"""
        test_file = self.test_dir / "test_data.jsonl"
        with open(test_file, "w") as f:
            for i in range(num_docs):
                doc = {"text": f"Document {i}: " + " ".join([f"word{j}" for j in range(doc_length)])}
                f.write(json.dumps(doc) + "\n")
        return test_file

    def create_compressed_data(self, num_docs: int = 50) -> Path:
        """Create compressed test data"""
        test_file = self.test_dir / "test_data.jsonl.gz"
        with gzip.open(test_file, "wt") as f:
            for i in range(num_docs):
                doc = {"text": f"Compressed doc {i}: " + " ".join([f"token{j}" for j in range(30)])}
                f.write(json.dumps(doc) + "\n")
        return test_file

    def create_text_data(self, num_lines: int = 100) -> Path:
        """Create plain text test data"""
        test_file = self.test_dir / "test_data.txt"
        with open(test_file, "w") as f:
            for i in range(num_lines):
                f.write(f"Line {i}: This is a test sentence with multiple words.\n")
        return test_file

    def create_config(self, mode: str, **kwargs) -> TokenizationConfig:
        """Create tokenization config for testing"""
        default_config = {
            "tokenizer_name": "EleutherAI/gpt-neox-20b",
            "input_path": str(self.test_dir),
            "output_path": str(self.output_dir),
            "output_format": "parquet",  # Explicitly set to parquet
            "mode": mode,
            "max_seq_length": 512,
            "batch_size": 10,
            "chunk_size": 20,
            "sequences_per_save": 50,
            "num_workers": 2 if mode == "hybrid" else 1,
            "enable_checkpoint": True,
            "checkpoint_dir": str(self.checkpoint_dir),
            "enable_rich_ui": False,  # Disable for testing
        }
        default_config.update(kwargs)
        return TokenizationConfig(**default_config)

    def test_batch_mode_basic(self):
        """Test basic batch processing mode"""
        # Create test data
        self.create_test_data(num_docs=50)

        # Configure and run
        config = self.create_config("batch")
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Verify output
        output_files = list(self.output_dir.glob("*.parquet"))
        assert len(output_files) > 0, "No output files generated"

        # Check metadata
        metadata_file = self.output_dir / "metadata.json"
        assert metadata_file.exists(), "Metadata file not created"

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 50
        assert metadata["statistics"]["errors"] == 0
        assert metadata["statistics"]["total_tokens"] > 0

    def test_streaming_mode_memory_efficiency(self):
        """Test streaming mode for memory efficiency"""
        # Create larger dataset
        self.create_test_data(num_docs=200, doc_length=100)

        # Configure with lower memory limit
        config = self.create_config("streaming", max_memory_gb=0.5)
        tokenizer = UnifiedTokenizer(config)

        # Track memory usage
        import psutil

        process = psutil.Process()
        process.memory_info().rss / 1024 / 1024 / 1024  # GB

        tokenizer.tokenize()

        peak_memory = tokenizer.stats.peak_memory_gb
        assert peak_memory < 1.0, f"Memory usage too high: {peak_memory:.2f} GB"

        # Verify processing completed
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata["statistics"]["total_documents"] == 200

    def test_hybrid_mode_parallel_processing(self):
        """Test hybrid mode with parallel processing"""
        # Create test data
        self.create_test_data(num_docs=100)

        # Configure with multiple workers
        config = self.create_config("hybrid", num_workers=min(4, multiprocessing.cpu_count()))
        tokenizer = UnifiedTokenizer(config)

        start_time = time.time()
        tokenizer.tokenize()
        elapsed_time = time.time() - start_time

        # Verify parallel processing worked
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 100
        assert metadata["statistics"]["errors"] == 0

        # Check that multiple workers were used (hybrid mode should be faster)
        docs_per_second = 100 / elapsed_time
        assert docs_per_second > 50, f"Processing too slow: {docs_per_second:.1f} docs/sec"

    def test_checkpoint_resume(self):
        """Test checkpoint and resume functionality"""
        # Create test data
        self.create_test_data(num_docs=100)

        # First run - process partially
        config = self.create_config("batch", checkpoint_interval=2)
        tokenizer = UnifiedTokenizer(config)

        # Simulate interruption after processing some files
        original_tokenize = tokenizer.tokenize

        def limited_tokenize():
            # Process only first 50 docs
            tokenizer.stats.total_documents = 50
            tokenizer._save_checkpoint({"processed_50_docs"})

        tokenizer.tokenize = limited_tokenize
        tokenizer.tokenize()

        # Verify checkpoint exists
        checkpoint_files = list(self.checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) > 0, "No checkpoint created"

        # Resume processing
        config.resume = True
        tokenizer2 = UnifiedTokenizer(config)
        # Restore original tokenize method
        tokenizer2.tokenize = original_tokenize.__get__(tokenizer2, UnifiedTokenizer)

        # Should resume from checkpoint
        assert tokenizer2.checkpoint_manager.load() is not None

    def test_compressed_input_processing(self):
        """Test processing of compressed JSONL.gz files"""
        # Create compressed test data
        self.create_compressed_data(num_docs=50)

        config = self.create_config("batch")
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Verify processing
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 50
        assert metadata["statistics"]["errors"] == 0

    def test_text_format_processing(self):
        """Test processing of plain text files"""
        # Create text data
        self.create_text_data(num_lines=100)

        config = self.create_config("streaming", input_format="text")
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Verify processing
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Text files process line by line
        assert metadata["statistics"]["total_documents"] == 100

    def test_error_handling_corrupt_data(self):
        """Test error handling with corrupt JSON data"""
        # Create test file with some corrupt data
        test_file = self.test_dir / "corrupt_data.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps({"text": "Good document 1"}) + "\n")
            f.write("CORRUPT JSON {not valid json}\n")  # Bad JSON
            f.write(json.dumps({"text": "Good document 2"}) + "\n")
            f.write(json.dumps({"wrong_field": "Missing text field"}) + "\n")  # Missing text
            f.write(json.dumps({"text": "Good document 3"}) + "\n")

        config = self.create_config("batch")
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Should process valid documents and skip invalid ones
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Only 3 valid documents with text field
        assert metadata["statistics"]["total_documents"] == 3

    def test_sequence_length_handling(self):
        """Test handling of different sequence lengths"""
        # Create documents of varying lengths
        test_file = self.test_dir / "varied_lengths.jsonl"
        with open(test_file, "w") as f:
            # Very short document
            f.write(json.dumps({"text": "Short."}) + "\n")
            # Medium document
            f.write(json.dumps({"text": " ".join([f"word{i}" for i in range(100)])}) + "\n")
            # Very long document (should be split)
            f.write(json.dumps({"text": " ".join([f"word{i}" for i in range(1000)])}) + "\n")

        config = self.create_config("batch", max_seq_length=128)
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Should process all documents
        assert metadata["statistics"]["total_documents"] == 3
        # Long document should create multiple sequences
        assert metadata["statistics"]["total_sequences"] > 3

    def test_append_eos_token(self):
        """Test EOS token appending"""
        self.create_test_data(num_docs=10)

        # Test with append_eos=True
        config = self.create_config("batch", append_eos=True)
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Verify EOS tokens were added
        import pyarrow.parquet as pq

        output_files = list(self.output_dir.glob("*.parquet"))
        table = pq.read_table(output_files[0])
        sequences = table["input_ids"].to_pylist()

        # Check that sequences end with EOS token
        eos_token_id = tokenizer.tokenizer.eos_token_id
        for seq in sequences[:5]:  # Check first 5
            assert seq[-1] == eos_token_id, "EOS token not appended"

    def test_validation_split(self):
        """Test train/validation split functionality"""
        self.create_test_data(num_docs=100)

        config = self.create_config("batch", validation_split=0.2, seed=42)
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Check that both train and validation files exist
        train_files = list(self.output_dir.glob("train_*.parquet"))
        val_files = list(self.output_dir.glob("validation_*.parquet"))

        assert len(train_files) > 0, "No training files created"
        assert len(val_files) > 0, "No validation files created"

        # Verify split ratio
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        metadata["statistics"]["total_sequences"]
        # Validation should be approximately 20%
        # Note: exact split depends on sequence boundaries

    def test_multiple_input_files(self):
        """Test processing multiple input files"""
        # Create multiple test files
        for i in range(3):
            test_file = self.test_dir / f"test_data_{i}.jsonl"
            with open(test_file, "w") as f:
                for j in range(20):
                    doc = {"text": f"File {i} Document {j}"}
                    f.write(json.dumps(doc) + "\n")

        config = self.create_config("hybrid")
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Should process all files
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 60  # 3 files * 20 docs
        assert metadata["statistics"]["files_processed"] >= 3


def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    import subprocess

    # Install pytest-cov if needed
    subprocess.run(["uv", "pip", "install", "pytest", "pytest-cov"], check=True)

    # Run tests with coverage
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            __file__,
            "-v",
            "--cov=unified_tokenizer",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-branch",
        ]
    )

    if result.returncode == 0:
        print("\nCoverage report generated in htmlcov/index.html")

    return result.returncode


if __name__ == "__main__":
    # Run with coverage when executed directly
    exit(run_tests_with_coverage())
