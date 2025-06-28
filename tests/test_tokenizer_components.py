"""
Component tests for tokenization system modules.
Tests individual components like configs, stats, checkpoints, and monitoring.
"""

import json
import sys
import time
from unittest.mock import Mock, patch

import pytest

# Import directly from scripts
sys.path.insert(0, "/home/gmatlin/Codespace/DataDecider/data_decide/scripts")
from unified_tokenizer import (
    CheckpointManager,
    MonitoringMixin,
    TokenizationConfig,
    TokenizationStats,
    _batch_tokenize_texts_standalone,
    _process_file_worker_standalone,
)


class TestTokenizationConfig:
    """Test TokenizationConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TokenizationConfig(input_path="/tmp/input", output_path="/tmp/output")

        assert config.tokenizer_name == "EleutherAI/gpt-neox-20b"
        assert config.max_seq_length == 2048
        assert config.mode == "hybrid"
        assert config.num_workers == 1
        assert config.enable_checkpoint is True

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = TokenizationConfig(input_path="/tmp/input", output_path="/tmp/output", mode="batch", num_workers=4)

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["input_path"] == "/tmp/input"
        assert config_dict["mode"] == "batch"
        assert config_dict["num_workers"] == 4

    def test_config_validation_modes(self):
        """Test configuration with different modes."""
        for mode in ["batch", "streaming", "hybrid"]:
            config = TokenizationConfig(input_path="/tmp", output_path="/tmp", mode=mode)
            assert config.mode == mode

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = TokenizationConfig(
            input_path="/custom/input",
            output_path="/custom/output",
            tokenizer_name="gpt2",
            max_seq_length=1024,
            batch_size=500,
            validation_split=0.1,
            compression="gzip",
        )

        assert config.tokenizer_name == "gpt2"
        assert config.max_seq_length == 1024
        assert config.batch_size == 500
        assert config.validation_split == 0.1
        assert config.compression == "gzip"


class TestTokenizationStats:
    """Test TokenizationStats tracking."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = TokenizationStats()

        assert stats.total_documents == 0
        assert stats.total_tokens == 0
        assert stats.total_sequences == 0
        assert stats.files_processed == 0
        assert stats.errors == 0
        assert stats.bytes_processed == 0
        assert stats.peak_memory_gb == 0.0
        assert stats.start_time > 0

    def test_stats_update_memory(self):
        """Test memory tracking."""
        stats = TokenizationStats()

        # Update memory
        stats.update_memory()
        assert stats.peak_memory_gb > 0

        # Peak should increase or stay same
        initial_peak = stats.peak_memory_gb
        stats.update_memory()
        assert stats.peak_memory_gb >= initial_peak

    def test_stats_get_rate(self):
        """Test rate calculations."""
        stats = TokenizationStats()
        stats.start_time = time.time() - 10  # 10 seconds ago
        stats.total_documents = 100
        stats.total_tokens = 5000

        docs_per_sec, tokens_per_sec = stats.get_rate()

        assert abs(docs_per_sec - 10.0) < 1.0  # ~10 docs/sec
        assert abs(tokens_per_sec - 500.0) < 10.0  # ~500 tokens/sec

    def test_stats_zero_elapsed_time(self):
        """Test rate calculation with zero elapsed time."""
        stats = TokenizationStats()
        stats.start_time = time.time()  # Just now

        docs_per_sec, tokens_per_sec = stats.get_rate()

        assert docs_per_sec == 0.0
        assert tokens_per_sec == 0.0


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_checkpoint_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        assert checkpoint_dir.exists()
        assert manager.checkpoint_file.name == "progress.json"

    def test_checkpoint_save_load(self, temp_dir):
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(str(temp_dir))

        # Create test data
        completed_files = ["file1.jsonl", "file2.jsonl"]
        stats = TokenizationStats()
        stats.total_documents = 100
        stats.total_tokens = 5000

        # Save checkpoint
        manager.save(completed_files, stats)

        # Load checkpoint
        checkpoint = manager.load()

        assert checkpoint is not None
        assert checkpoint["completed_files"] == completed_files
        assert checkpoint["stats"]["total_documents"] == 100
        assert checkpoint["stats"]["total_tokens"] == 5000

    def test_checkpoint_load_nonexistent(self, temp_dir):
        """Test loading non-existent checkpoint."""
        manager = CheckpointManager(str(temp_dir / "nonexistent"))
        checkpoint = manager.load()

        assert checkpoint is None

    def test_checkpoint_get_completed_files(self, temp_dir):
        """Test getting completed files from checkpoint."""
        manager = CheckpointManager(str(temp_dir))

        # No checkpoint yet
        files = manager.get_completed_files()
        assert files == []

        # Save checkpoint
        completed_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]
        manager.save(completed_files, TokenizationStats())

        # Get completed files
        files = manager.get_completed_files()
        assert files == completed_files

    def test_checkpoint_atomic_write(self, temp_dir):
        """Test atomic checkpoint writing."""
        manager = CheckpointManager(str(temp_dir))

        # Save checkpoint
        manager.save(["file1.jsonl"], TokenizationStats())

        # Verify temp file doesn't exist (atomic rename)
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0

        # Verify checkpoint exists
        assert manager.checkpoint_file.exists()


class TestMonitoringMixin:
    """Test MonitoringMixin functionality."""

    def test_monitoring_initialization(self, base_config):
        """Test monitoring mixin initialization."""
        config = TokenizationConfig(**base_config)

        class TestMonitor(MonitoringMixin):
            def __init__(self, config):
                super().__init__(config)

        monitor = TestMonitor(config)

        assert monitor.config == config
        assert isinstance(monitor.stats, TokenizationStats)
        assert monitor.progress_bar is None
        assert monitor.live_display is None

    def test_monitoring_update_stats(self, base_config):
        """Test stats updating through monitoring."""
        config = TokenizationConfig(**base_config)

        class TestMonitor(MonitoringMixin):
            def __init__(self, config):
                super().__init__(config)

        monitor = TestMonitor(config)

        # Update stats
        monitor.update_monitoring(total_documents=10, total_tokens=500, files_processed=1)

        assert monitor.stats.total_documents == 10
        assert monitor.stats.total_tokens == 500
        assert monitor.stats.files_processed == 1

    @patch("unified_tokenizer.tqdm")
    def test_monitoring_progress_bar(self, mock_tqdm, base_config):
        """Test progress bar monitoring."""
        config = TokenizationConfig(**base_config)
        config.enable_rich_ui = False  # Use tqdm

        class TestMonitor(MonitoringMixin):
            def __init__(self, config):
                super().__init__(config)

        monitor = TestMonitor(config)

        # Start monitoring
        monitor.start_monitoring(total_files=10)

        # Should create tqdm progress bar
        mock_tqdm.assert_called_once_with(total=10, desc="Processing files")
        assert monitor.progress_bar is not None

        # Update monitoring
        monitor.update_monitoring(files_processed=1)
        monitor.progress_bar.update.assert_called_with(1)

        # Stop monitoring
        monitor.stop_monitoring()
        monitor.progress_bar.close.assert_called_once()


class TestStandaloneWorkerFunctions:
    """Test standalone worker functions for parallel processing."""

    def test_batch_tokenize_texts_standalone(self):
        """Test standalone batch tokenization function."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.return_value = {
            "input_ids": [
                [1, 2, 3, 4, 5],  # Short sequence
                list(range(100)),  # Long sequence
                list(range(600)),  # Very long sequence
            ]
        }

        # Mock config
        mock_config = Mock()
        mock_config.max_seq_length = 512
        mock_config.append_eos = True

        texts = ["short text", "medium text", "long text"]

        sequences = _batch_tokenize_texts_standalone(texts, mock_tokenizer, mock_config)

        # Should process all sequences appropriately
        # Note: The actual behavior depends on the min threshold (10% of 512 = 51.2)
        # So sequence with 5 tokens is filtered, 100 tokens fits in one, 600 tokens splits into 2
        assert len(sequences) == 3  # Short filtered, medium kept, long split

    def test_process_file_worker_standalone(self, temp_dir, create_jsonl_file):
        """Test standalone file processing worker."""
        # Create test file
        data = [{"text": f"Document {i}"} for i in range(10)]
        file_path = create_jsonl_file("worker_test.jsonl", data)

        # Mock config
        mock_config = Mock()
        mock_config.tokenizer_name = "EleutherAI/gpt-neox-20b"
        mock_config.chunk_size = 5
        mock_config.max_seq_length = 512
        mock_config.append_eos = True

        # Mock the tokenizer loading to avoid network calls
        with patch("unified_tokenizer.AutoTokenizer.from_pretrained") as mock_tokenizer:
            # Setup mock tokenizer
            tokenizer_instance = Mock()
            tokenizer_instance.eos_token_id = 50256
            tokenizer_instance.return_value = {"input_ids": [[1, 2, 3, 4, 5] for _ in range(5)]}
            mock_tokenizer.return_value = tokenizer_instance

            # Process file
            sequences, stats = _process_file_worker_standalone((file_path, mock_config))

            # Verify results
            assert isinstance(sequences, list)
            assert isinstance(stats, dict)
            assert stats["total_documents"] == 10
            assert stats["files_processed"] == 1
            assert stats["bytes_processed"] == file_path.stat().st_size

    def test_process_file_worker_error_handling(self, temp_dir):
        """Test worker error handling."""
        # Create corrupt file
        file_path = temp_dir / "corrupt.jsonl"
        with open(file_path, "w") as f:
            f.write("{invalid json}\n")

        mock_config = Mock()
        mock_config.tokenizer_name = "EleutherAI/gpt-neox-20b"
        mock_config.chunk_size = 5

        with patch("unified_tokenizer.AutoTokenizer.from_pretrained"):
            # Should handle the error gracefully
            sequences, stats = _process_file_worker_standalone((file_path, mock_config))

            # Should return empty results
            assert sequences == []
            assert stats["total_documents"] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_config_validation(self):
        """Test configuration with minimal required fields."""
        # Should not raise error
        config = TokenizationConfig(input_path="", output_path="")
        assert config.input_path == ""
        assert config.output_path == ""

    def test_stats_with_no_time_elapsed(self):
        """Test stats when no time has elapsed."""
        stats = TokenizationStats()
        stats.start_time = time.time()

        # Immediately get rate
        docs_rate, tokens_rate = stats.get_rate()

        assert docs_rate == 0.0
        assert tokens_rate == 0.0

    def test_checkpoint_corrupted_file(self, temp_dir):
        """Test handling corrupted checkpoint file."""
        manager = CheckpointManager(str(temp_dir))

        # Create corrupted checkpoint
        with open(manager.checkpoint_file, "w") as f:
            f.write("{invalid json content")

        # Should raise JSONDecodeError (current behavior)
        # or could be modified to handle gracefully
        with pytest.raises(json.JSONDecodeError):
            checkpoint = manager.load()

        # Should still be able to save new checkpoint
        manager.save(["file1.jsonl"], TokenizationStats())

        # New checkpoint should be valid
        checkpoint = manager.load()
        assert checkpoint is not None
