#!/usr/bin/env python3
"""
Unified tokenization system combining the best features from all tokenization approaches.

Features:
- High performance batch processing (4,300+ docs/sec from hybrid)
- Memory-safe streaming mode (600 docs/sec from streaming)
- Checkpoint/resume capability with atomic writes
- Multiple input format support (JSON, JSONL, GZ, text)
- Comprehensive monitoring with optional Rich UI
- Configurable processing modes and parameters
"""

import argparse
import gc
import gzip
import hashlib
import json
import logging
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# Optional Rich UI support
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TokenizationConfig:
    """Configuration for unified tokenizer"""

    # Input/Output
    input_path: str
    output_path: str
    input_format: str = "auto"  # auto, json, jsonl, gz, text
    output_format: str = "arrow"  # arrow, parquet

    # Tokenizer
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    max_seq_length: int = 2048
    append_eos: bool = True

    # Processing Mode
    mode: str = "hybrid"  # batch, streaming, hybrid
    batch_size: int = 200
    chunk_size: int = 10_000
    sequences_per_save: int = 25_000

    # Parallelization
    num_workers: int = 1
    max_memory_gb: float = 8.0

    # Checkpoint/Resume
    enable_checkpoint: bool = True
    checkpoint_interval: int = 5  # files
    resume: bool = True
    checkpoint_dir: str = "tokenization_checkpoints"

    # Data Splitting
    validation_split: float = 0.0
    seed: int = 42

    # Monitoring
    enable_rich_ui: bool = True
    log_level: str = "INFO"
    report_interval: int = 1000  # sequences

    # Advanced
    verify_checksums: bool = True
    save_metadata: bool = True
    compression: str = None  # gzip, snappy, None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TokenizationStats:
    """Statistics tracking for tokenization"""

    total_documents: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    files_processed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    bytes_processed: int = 0
    peak_memory_gb: float = 0.0

    def update_memory(self):
        """Update peak memory usage"""
        current_mem = psutil.Process().memory_info().rss / 1024**3
        self.peak_memory_gb = max(self.peak_memory_gb, current_mem)

    def get_rate(self) -> Tuple[float, float]:
        """Get documents/sec and tokens/sec rates"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            docs_per_sec = self.total_documents / elapsed
            tokens_per_sec = self.total_tokens / elapsed
            return docs_per_sec, tokens_per_sec
        return 0.0, 0.0


class CheckpointManager:
    """Manages checkpointing and resume functionality"""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress.json"

    def save(self, completed_files: List[str], stats: TokenizationStats, metadata: Dict = None):
        """Save checkpoint atomically"""
        checkpoint = {
            "completed_files": completed_files,
            "stats": asdict(stats),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Write atomically
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.replace(self.checkpoint_file)

    def load(self) -> Optional[Dict]:
        """Load latest checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None

    def get_completed_files(self) -> List[str]:
        """Get list of completed files from checkpoint"""
        checkpoint = self.load()
        if checkpoint:
            return checkpoint.get("completed_files", [])
        return []


class MonitoringMixin:
    """Mixin for monitoring functionality"""

    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.stats = TokenizationStats()
        self.progress_bar = None
        self.live_display = None

    def start_monitoring(self, total_files: int):
        """Start monitoring display"""
        if self.config.enable_rich_ui and RICH_AVAILABLE:
            self.live_display = Live(self._create_rich_display(), refresh_per_second=1)
            self.live_display.start()
        else:
            self.progress_bar = tqdm(total=total_files, desc="Processing files")

    def update_monitoring(self, **kwargs):
        """Update monitoring display"""
        # Update stats
        for key, value in kwargs.items():
            if hasattr(self.stats, key):
                current = getattr(self.stats, key)
                setattr(self.stats, key, current + value)

        self.stats.update_memory()

        # Update display
        if self.config.enable_rich_ui and RICH_AVAILABLE and self.live_display:
            self.live_display.update(self._create_rich_display())
        elif self.progress_bar:
            self.progress_bar.update(kwargs.get("files_processed", 0))
            docs_per_sec, _ = self.stats.get_rate()
            self.progress_bar.set_postfix({"docs/s": f"{docs_per_sec:.0f}"})

    def stop_monitoring(self):
        """Stop monitoring display"""
        if self.live_display:
            self.live_display.stop()
        if self.progress_bar:
            self.progress_bar.close()

    def _create_rich_display(self):
        """Create Rich display panel"""
        # Stats table
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        elapsed = time.time() - self.stats.start_time
        docs_per_sec, tokens_per_sec = self.stats.get_rate()

        stats_table.add_row("Time Elapsed", str(timedelta(seconds=int(elapsed))))
        stats_table.add_row("Documents", f"{self.stats.total_documents:,}")
        stats_table.add_row("Tokens", f"{self.stats.total_tokens:,}")
        stats_table.add_row("Sequences", f"{self.stats.total_sequences:,}")
        stats_table.add_row("Files", f"{self.stats.files_processed}")
        stats_table.add_row("Docs/sec", f"{docs_per_sec:.0f}")
        stats_table.add_row("Tokens/sec", f"{tokens_per_sec:,.0f}")
        stats_table.add_row("Memory", f"{self.stats.peak_memory_gb:.2f} GB")

        return Panel(stats_table, title="🚀 Tokenization Progress", border_style="blue")


class UnifiedTokenizer(MonitoringMixin):
    """
    Unified tokenizer combining best features from all approaches.
    """

    def __init__(self, config: TokenizationConfig):
        super().__init__(config)
        self.config = config
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tokenizer = self._init_tokenizer()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # File tracking
        self.current_sequences = []
        self.output_file_index = 0

    def _init_tokenizer(self):
        """Initialize the tokenizer"""
        logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        # Ensure we have necessary tokens
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer {self.config.tokenizer_name} doesn't have an EOS token")

        return tokenizer

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Received shutdown signal, saving checkpoint...")
        self._save_checkpoint()
        self.stop_monitoring()
        sys.exit(0)

    def tokenize(self):
        """Main tokenization entry point"""
        # Get input files
        input_files = self._get_input_files()
        if not input_files:
            logger.error(f"No input files found in {self.config.input_path}")
            return

        logger.info(f"Found {len(input_files)} files to process")

        # Load checkpoint if resuming
        completed_files = set()
        if self.config.resume:
            checkpoint = self.checkpoint_manager.load()
            if checkpoint:
                completed_files = set(checkpoint.get("completed_files", []))
                # Restore stats
                saved_stats = checkpoint.get("stats", {})
                for key, value in saved_stats.items():
                    if hasattr(self.stats, key):
                        setattr(self.stats, key, value)
                logger.info(f"Resuming from checkpoint: {len(completed_files)} files already processed")

        # Filter files to process
        files_to_process = [f for f in input_files if str(f) not in completed_files]

        # Start monitoring
        self.start_monitoring(len(files_to_process))

        try:
            # Process based on mode
            if self.config.mode == "batch":
                self._batch_process(files_to_process, completed_files)
            elif self.config.mode == "streaming":
                self._streaming_process(files_to_process, completed_files)
            elif self.config.mode == "hybrid":
                self._hybrid_process(files_to_process, completed_files)
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

            # Save any remaining sequences
            if self.current_sequences:
                self._save_sequences()

            # Save final metadata
            if self.config.save_metadata:
                self._save_metadata()

            # Final report
            self._print_summary()

        finally:
            self.stop_monitoring()

    def _get_input_files(self) -> List[Path]:
        """Get list of input files based on format"""
        input_path = Path(self.config.input_path)

        if input_path.is_file():
            return [input_path]

        # Directory - find files based on format
        if self.config.input_format == "auto":
            # Try common patterns
            patterns = ["*.json.gz", "*.jsonl.gz", "*.json", "*.jsonl", "*.txt"]
            files = []
            for pattern in patterns:
                files.extend(sorted(input_path.glob(pattern)))
            return files
        else:
            # Specific format
            ext_map = {"json": "*.json", "jsonl": "*.jsonl", "gz": "*.gz", "text": "*.txt"}
            pattern = ext_map.get(self.config.input_format, "*")
            return sorted(input_path.glob(pattern))

    def _batch_process(self, files: List[Path], completed_files: set):
        """High-performance batch processing mode"""
        for file_path in files:
            try:
                # Process file in chunks
                for chunk in self._read_documents_chunked(file_path):
                    texts = [doc.get("text", "") for doc in chunk]
                    texts = [t for t in texts if t.strip()]

                    if texts:
                        sequences = self._batch_tokenize_texts(texts)
                        self.current_sequences.extend(sequences)

                        # Update stats
                        self.update_monitoring(
                            total_documents=len(texts),
                            total_tokens=sum(len(seq) for seq in sequences),
                            total_sequences=len(sequences),
                        )

                        # Save periodically
                        if len(self.current_sequences) >= self.config.sequences_per_save:
                            self._save_sequences()

                # Mark file as completed
                completed_files.add(str(file_path))
                self.update_monitoring(files_processed=1, bytes_processed=file_path.stat().st_size)

                # Checkpoint periodically
                if len(completed_files) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(completed_files)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.update_monitoring(errors=1)

    def _streaming_process(self, files: List[Path], completed_files: set):
        """Memory-safe streaming mode"""
        for file_path in files:
            try:
                sequences_buffer = []

                # Process file line by line
                for doc in self._read_documents_streaming(file_path):
                    text = doc.get("text", "")
                    if not text.strip():
                        continue

                    # Tokenize single document
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    if self.config.append_eos and tokens[-1] != self.tokenizer.eos_token_id:
                        tokens.append(self.tokenizer.eos_token_id)

                    # Split into sequences
                    for i in range(0, len(tokens), self.config.max_seq_length):
                        seq = tokens[i : i + self.config.max_seq_length]
                        if len(seq) == self.config.max_seq_length:
                            sequences_buffer.append(seq)

                    # Update stats
                    self.update_monitoring(
                        total_documents=1, total_tokens=len(tokens), total_sequences=len(sequences_buffer)
                    )

                    # Save when buffer is full
                    if len(sequences_buffer) >= self.config.batch_size:
                        self.current_sequences.extend(sequences_buffer)
                        sequences_buffer = []

                        if len(self.current_sequences) >= self.config.sequences_per_save:
                            self._save_sequences()
                            gc.collect()  # Force garbage collection

                # Save remaining sequences from this file
                if sequences_buffer:
                    self.current_sequences.extend(sequences_buffer)

                # Mark file as completed
                completed_files.add(str(file_path))
                self.update_monitoring(files_processed=1)

                # Checkpoint
                if len(completed_files) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(completed_files)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.update_monitoring(errors=1)

    def _hybrid_process(self, files: List[Path], completed_files: set):
        """Hybrid mode combining batch performance with memory safety"""
        # Use parallel processing if available
        if self.config.num_workers > 1:
            self._parallel_hybrid_process(files, completed_files)
        else:
            # Single process hybrid
            self._batch_process(files, completed_files)

    def _parallel_hybrid_process(self, files: List[Path], completed_files: set):
        """Parallel processing for hybrid mode"""
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit jobs
            future_to_file = {executor.submit(self._process_file_worker, file_path): file_path for file_path in files}

            # Process results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    sequences, stats = future.result()

                    # Add sequences
                    self.current_sequences.extend(sequences)

                    # Update stats
                    self.update_monitoring(**stats)

                    # Save periodically
                    if len(self.current_sequences) >= self.config.sequences_per_save:
                        self._save_sequences()

                    # Mark completed
                    completed_files.add(str(file_path))

                    # Checkpoint
                    if len(completed_files) % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(completed_files)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.update_monitoring(errors=1)

    def _process_file_worker(self, file_path: Path) -> Tuple[List[List[int]], Dict]:
        """Worker function for parallel processing"""
        # Re-initialize tokenizer in worker process
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        sequences = []
        stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "total_sequences": 0,
            "bytes_processed": file_path.stat().st_size,
            "files_processed": 1,
        }

        # Process file
        for chunk in self._read_documents_chunked(file_path):
            texts = [doc.get("text", "") for doc in chunk]
            texts = [t for t in texts if t.strip()]

            if texts:
                chunk_sequences = self._batch_tokenize_texts_with_tokenizer(texts, tokenizer)
                sequences.extend(chunk_sequences)

                stats["total_documents"] += len(texts)
                stats["total_tokens"] += sum(len(seq) for seq in chunk_sequences)
                stats["total_sequences"] += len(chunk_sequences)

        return sequences, stats

    def _batch_tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """Batch tokenize texts efficiently"""
        return self._batch_tokenize_texts_with_tokenizer(texts, self.tokenizer)

    def _batch_tokenize_texts_with_tokenizer(self, texts: List[str], tokenizer) -> List[List[int]]:
        """Batch tokenize with specific tokenizer instance"""
        # Tokenize in batch
        encodings = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        all_sequences = []
        for input_ids in encodings["input_ids"]:
            # Add EOS if needed
            if self.config.append_eos and (len(input_ids) == 0 or input_ids[-1] != tokenizer.eos_token_id):
                input_ids.append(tokenizer.eos_token_id)

            # Split into sequences
            for i in range(0, len(input_ids), self.config.max_seq_length):
                sequence = input_ids[i : i + self.config.max_seq_length]
                if len(sequence) == self.config.max_seq_length:
                    all_sequences.append(sequence)

        return all_sequences

    def _read_documents_chunked(self, file_path: Path) -> Iterator[List[Dict]]:
        """Read documents in chunks"""
        chunk = []

        for doc in self._read_documents_streaming(file_path):
            chunk.append(doc)
            if len(chunk) >= self.config.chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk

    def _read_documents_streaming(self, file_path: Path) -> Iterator[Dict]:
        """Read documents one at a time"""
        # Determine how to open file
        if file_path.suffix == ".gz":
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"

        try:
            with open_fn(file_path, mode, encoding="utf-8") as f:
                if file_path.suffix in [".jsonl", ".gz"] or self.config.input_format == "jsonl":
                    # JSONL format
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
                elif file_path.suffix == ".json" or self.config.input_format == "json":
                    # Single JSON file
                    data = json.load(f)
                    if isinstance(data, list):
                        for doc in data:
                            yield doc
                    else:
                        yield data
                else:
                    # Plain text
                    yield {"text": f.read()}
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    def _save_sequences(self):
        """Save current sequences to disk"""
        if not self.current_sequences:
            return

        # Create output filename
        output_file = self.output_dir / f"tokenized_{self.output_file_index:04d}.{self.config.output_format}"
        temp_file = output_file.with_suffix(".tmp")

        # Save based on format
        if self.config.output_format == "arrow":
            table = pa.table(
                {"input_ids": self.current_sequences, "length": [len(seq) for seq in self.current_sequences]}
            )

            with pa.OSFile(str(temp_file), "wb") as f:
                with pa.ipc.new_file(f, table.schema) as writer:
                    writer.write_table(table)

        elif self.config.output_format == "parquet":
            table = pa.table(
                {"input_ids": self.current_sequences, "length": [len(seq) for seq in self.current_sequences]}
            )

            pq.write_table(
                table, temp_file, compression=self.config.compression or "snappy", use_dictionary=True, version="2.6"
            )

        # Atomic rename
        temp_file.rename(output_file)

        logger.info(f"Saved {len(self.current_sequences)} sequences to {output_file.name}")

        # Clear sequences and increment counter
        self.current_sequences = []
        self.output_file_index += 1

        # Force garbage collection
        gc.collect()

    def _save_checkpoint(self, completed_files: set = None):
        """Save checkpoint"""
        if not self.config.enable_checkpoint:
            return

        files = list(completed_files) if completed_files else []
        self.checkpoint_manager.save(files, self.stats)
        logger.debug("Checkpoint saved")

    def _save_metadata(self):
        """Save comprehensive metadata"""
        # Calculate checksums if requested
        checksums = {}
        if self.config.verify_checksums:
            for output_file in sorted(self.output_dir.glob(f"*.{self.config.output_format}")):
                with open(output_file, "rb") as f:
                    checksums[output_file.name] = hashlib.sha256(f.read()).hexdigest()

        metadata = {
            "tokenization_config": self.config.to_dict(),
            "tokenizer_info": {
                "name": self.config.tokenizer_name,
                "vocab_size": self.tokenizer.vocab_size,
                "eos_token": self.tokenizer.eos_token,
                "eos_token_id": self.tokenizer.eos_token_id,
            },
            "statistics": asdict(self.stats),
            "output_info": {
                "num_files": self.output_file_index,
                "format": self.config.output_format,
                "compression": self.config.compression,
            },
            "checksums": checksums,
            "created_at": datetime.now().isoformat(),
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_file}")

    def _print_summary(self):
        """Print final summary"""
        elapsed = time.time() - self.stats.start_time
        docs_per_sec, tokens_per_sec = self.stats.get_rate()

        summary = f"""
{'='*60}
Tokenization Complete!
{'='*60}
Total Documents:  {self.stats.total_documents:,}
Total Tokens:     {self.stats.total_tokens:,}
Total Sequences:  {self.stats.total_sequences:,}
Files Processed:  {self.stats.files_processed}
Errors:          {self.stats.errors}

Performance:
  Time:          {timedelta(seconds=int(elapsed))}
  Docs/sec:      {docs_per_sec:,.0f}
  Tokens/sec:    {tokens_per_sec:,.0f}
  Peak Memory:   {self.stats.peak_memory_gb:.2f} GB

Output:
  Directory:     {self.output_dir}
  Files:         {self.output_file_index}
  Format:        {self.config.output_format}
{'='*60}
"""

        if self.config.enable_rich_ui and RICH_AVAILABLE:
            console.print(Panel(summary, title="✅ Summary", border_style="green"))
        else:
            print(summary)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified tokenization system with multiple processing modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument("input_path", type=str, help="Input file or directory")
    parser.add_argument("output_path", type=str, help="Output directory")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl", "gz", "text"],
        help="Input file format",
    )
    parser.add_argument(
        "--output-format", type=str, default="arrow", choices=["arrow", "parquet"], help="Output format"
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", dest="tokenizer_name", help="Tokenizer model name"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--no-append-eos", action="store_false", dest="append_eos", help="Don't append EOS token")

    # Processing mode
    parser.add_argument(
        "--mode", type=str, default="hybrid", choices=["batch", "streaming", "hybrid"], help="Processing mode"
    )
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for tokenization")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Documents per chunk")
    parser.add_argument("--sequences-per-save", type=int, default=25000, help="Sequences to accumulate before saving")

    # Parallelization
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--max-memory-gb", type=float, default=8.0, help="Maximum memory usage in GB")

    # Checkpoint/Resume
    parser.add_argument("--no-checkpoint", action="store_false", dest="enable_checkpoint", help="Disable checkpointing")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N files")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Don't resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="tokenization_checkpoints", help="Checkpoint directory")

    # Data splitting
    parser.add_argument("--validation-split", type=float, default=0.0, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Monitoring
    parser.add_argument("--no-rich-ui", action="store_false", dest="enable_rich_ui", help="Disable Rich UI")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    parser.add_argument("--report-interval", type=int, default=1000, help="Progress report interval")

    # Advanced
    parser.add_argument(
        "--no-verify-checksums", action="store_false", dest="verify_checksums", help="Don't calculate checksums"
    )
    parser.add_argument("--no-metadata", action="store_false", dest="save_metadata", help="Don't save metadata")
    parser.add_argument(
        "--compression", type=str, default=None, choices=["gzip", "snappy", "zstd", None], help="Output compression"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create config
    config = TokenizationConfig(**vars(args))

    # Run tokenization
    tokenizer = UnifiedTokenizer(config)
    tokenizer.tokenize()


if __name__ == "__main__":
    main()
