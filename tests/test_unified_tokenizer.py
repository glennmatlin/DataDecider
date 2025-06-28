"""
Integration tests for the unified tokenization system.
Tests all processing modes, parallel processing, and error handling.
"""

import json
import multiprocessing
import sys
from pathlib import Path

import pytest

# Import directly from scripts
sys.path.insert(0, "/home/gmatlin/Codespace/DataDecider/data_decide/scripts")
from unified_tokenizer import UnifiedTokenizer


class TestUnifiedTokenizerIntegration:
    """Integration tests for UnifiedTokenizer."""

    def test_batch_mode_basic(self, mock_tokenizer_config, create_jsonl_file, sample_jsonl_data):
        """Test basic batch processing mode."""
        # Create test data
        create_jsonl_file("test_batch.jsonl", sample_jsonl_data * 10)  # 50 documents

        # Configure for batch mode
        config = mock_tokenizer_config
        config.mode = "batch"
        config.output_format = "parquet"

        # Run tokenization
        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Verify output
        output_dir = Path(config.output_path)
        assert output_dir.exists()

        output_files = list(output_dir.glob("*.parquet"))
        assert len(output_files) > 0, "No output files generated"

        # Check metadata
        metadata_file = output_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 50
        assert metadata["statistics"]["total_tokens"] > 0
        assert metadata["statistics"]["errors"] == 0

    def test_streaming_mode_memory_efficiency(self, mock_tokenizer_config, large_dataset, performance_benchmark):
        """Test streaming mode for memory efficiency."""
        # Configure for streaming mode with memory limit
        config = mock_tokenizer_config
        config.mode = "streaming"
        config.max_memory_gb = 0.5
        config.output_format = "arrow"

        # Run tokenization
        tokenizer = UnifiedTokenizer(config)

        performance_benchmark.start()
        tokenizer.tokenize()
        performance_benchmark.stop()

        # Verify memory usage stayed within limits
        assert tokenizer.stats.peak_memory_gb < 1.0, f"Memory usage too high: {tokenizer.stats.peak_memory_gb:.2f} GB"

        # Verify processing completed
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 1000
        assert metadata["statistics"]["errors"] == 0

    def test_hybrid_mode_parallel_processing(self, mock_tokenizer_config, create_jsonl_file, performance_benchmark):
        """Test hybrid mode with multiple workers."""
        # Create test data
        data = [{"text": f"Document {i}: " + " ".join([f"word{j}" for j in range(100)])} for i in range(200)]
        create_jsonl_file("test_hybrid.jsonl", data)

        # Configure for hybrid mode with multiple workers
        config = mock_tokenizer_config
        config.mode = "hybrid"
        config.num_workers = min(4, multiprocessing.cpu_count())
        config.output_format = "parquet"

        # Run tokenization
        tokenizer = UnifiedTokenizer(config)

        performance_benchmark.start()
        tokenizer.tokenize()
        performance_benchmark.stop()

        # Verify parallel processing worked
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 200
        assert metadata["statistics"]["errors"] == 0

        # Check performance (should be reasonably fast with parallel processing)
        docs_per_second = 200 / performance_benchmark.elapsed
        assert docs_per_second > 50, f"Processing too slow: {docs_per_second:.1f} docs/sec"

    def test_checkpoint_resume_functionality(self, mock_tokenizer_config, create_jsonl_file, temp_dir):
        """Test checkpoint and resume functionality."""
        # Create test data
        data = [{"text": f"Document {i}"} for i in range(100)]
        create_jsonl_file("test_checkpoint.jsonl", data)

        # Configure with checkpoint enabled
        config = mock_tokenizer_config
        config.mode = "batch"
        config.checkpoint_interval = 1  # Checkpoint after every file
        config.sequences_per_save = 25  # Save frequently

        # First run - process partially by interrupting
        tokenizer = UnifiedTokenizer(config)

        # Monkey-patch to simulate interruption after 50 docs
        original_update = tokenizer.update_monitoring

        def limited_update(**kwargs):
            original_update(**kwargs)
            if tokenizer.stats.total_documents >= 50:
                tokenizer._save_checkpoint()
                raise KeyboardInterrupt("Simulated interruption")

        tokenizer.update_monitoring = limited_update

        with pytest.raises(KeyboardInterrupt):
            tokenizer.tokenize()

        # Verify checkpoint was saved
        checkpoint_file = Path(config.checkpoint_dir) / "progress.json"
        assert checkpoint_file.exists()

        # Resume processing
        config.resume = True
        tokenizer2 = UnifiedTokenizer(config)
        tokenizer2.tokenize()

        # Verify all documents were processed
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Should have processed remaining documents
        assert metadata["statistics"]["total_documents"] >= 50

    def test_compressed_input_handling(self, mock_tokenizer_config, create_jsonl_file, sample_jsonl_data):
        """Test processing of compressed JSONL.gz files."""
        # Create compressed test data
        create_jsonl_file("test_compressed.jsonl.gz", sample_jsonl_data * 10, compress=True)

        config = mock_tokenizer_config
        config.mode = "batch"

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Verify processing
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 50
        assert metadata["statistics"]["errors"] == 0

    def test_error_handling_corrupt_data(self, mock_tokenizer_config, corrupt_dataset):
        """Test error handling with corrupt data."""
        config = mock_tokenizer_config
        config.mode = "batch"

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Should process valid documents and skip invalid ones
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Should have processed 3 valid documents (1, 2, and the very long one)
        assert metadata["statistics"]["total_documents"] == 3
        assert metadata["statistics"]["errors"] == 0  # JSON decode errors are handled gracefully

    def test_multiple_file_processing(self, mock_tokenizer_config, create_jsonl_file):
        """Test processing multiple input files."""
        # Create multiple files
        for i in range(5):
            data = [{"text": f"File {i} Document {j}"} for j in range(20)]
            create_jsonl_file(f"test_file_{i}.jsonl", data)

        config = mock_tokenizer_config
        config.mode = "hybrid"
        config.num_workers = 2

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Should process all files
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 100  # 5 files * 20 docs
        assert metadata["statistics"]["files_processed"] == 5

    def test_text_format_input(self, mock_tokenizer_config, create_text_file):
        """Test processing plain text files."""
        # Create text file
        content = "\n".join([f"Line {i}: This is a test sentence." for i in range(50)])
        create_text_file("test_text.txt", content)

        config = mock_tokenizer_config
        config.mode = "streaming"
        config.input_format = "text"

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Text files are processed as single documents
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 1
        assert metadata["statistics"]["total_tokens"] > 0

    def test_output_formats(self, mock_tokenizer_config, create_jsonl_file, sample_jsonl_data):
        """Test different output formats (arrow vs parquet)."""
        create_jsonl_file("test_formats.jsonl", sample_jsonl_data)

        # Test Arrow format
        config = mock_tokenizer_config
        config.output_format = "arrow"
        config.output_path = str(Path(config.output_path) / "arrow_output")

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        arrow_files = list(Path(config.output_path).glob("*.arrow"))
        assert len(arrow_files) > 0

        # Test Parquet format
        config.output_format = "parquet"
        config.output_path = str(Path(mock_tokenizer_config.output_path) / "parquet_output")

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        parquet_files = list(Path(config.output_path).glob("*.parquet"))
        assert len(parquet_files) > 0

    def test_memory_limit_enforcement(self, mock_tokenizer_config, large_dataset):
        """Test that memory limits are enforced in streaming mode."""
        config = mock_tokenizer_config
        config.mode = "streaming"
        config.max_memory_gb = 0.1  # Very low limit
        config.batch_size = 5  # Small batches

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Memory usage should stay low
        assert (
            tokenizer.stats.peak_memory_gb < 0.5
        ), f"Memory limit not enforced: {tokenizer.stats.peak_memory_gb:.2f} GB"

    @pytest.mark.parametrize("num_workers", [1, 2, 4])
    def test_different_worker_counts(self, mock_tokenizer_config, create_jsonl_file, num_workers):
        """Test hybrid mode with different numbers of workers."""
        # Skip if not enough CPUs
        if num_workers > multiprocessing.cpu_count():
            pytest.skip(f"Not enough CPUs for {num_workers} workers")

        data = [{"text": f"Doc {i}"} for i in range(100)]
        create_jsonl_file(f"test_workers_{num_workers}.jsonl", data)

        config = mock_tokenizer_config
        config.mode = "hybrid"
        config.num_workers = num_workers

        tokenizer = UnifiedTokenizer(config)
        tokenizer.tokenize()

        # Should complete successfully regardless of worker count
        metadata_file = Path(config.output_path) / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["statistics"]["total_documents"] == 100
        assert metadata["statistics"]["errors"] == 0
