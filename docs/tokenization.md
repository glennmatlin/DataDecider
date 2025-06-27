# Tokenization Guide

This guide consolidates all tokenization documentation for the DataDecider project, covering architecture, performance, and usage.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Performance](#performance)
5. [Usage Guide](#usage-guide)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

## Overview

The DataDecider tokenization system provides a unified, high-performance solution for tokenizing large datasets. The system was consolidated from multiple implementations into a single, feature-rich tokenizer.

### Key Features
- **Multiple modes**: Batch (4,300+ docs/sec), Streaming (memory-safe), Hybrid
- **Checkpoint/resume**: Atomic writes for fault tolerance
- **Parallel processing**: Multi-worker support with ProcessPoolExecutor
- **Memory efficient**: Configurable memory limits and chunked processing
- **Rich monitoring**: Real-time progress with Rich UI
- **Format support**: JSON, JSONL, GZ, text files

## Quick Start

### Basic Usage
```bash
# Standard tokenization
python data_decide/scripts/tokenize.py /path/to/data output/dir

# With specific tokenizer
python data_decide/scripts/tokenize.py /path/to/data output/dir --tokenizer allenai/OLMo-1B

# Resume from checkpoint
python data_decide/scripts/tokenize.py /path/to/data output/dir --resume
```

### Count Tokens
```bash
# Sample-based estimation (default: 1000 docs)
python data_decide/scripts/count_tokens_unified.py

# Exact count of all tokens
python data_decide/scripts/count_tokens_unified.py --exact

# Custom sample size
python data_decide/scripts/count_tokens_unified.py --sample-size 5000
```

## Architecture

### Unified Tokenizer Design
The tokenization system follows a modular architecture:

```
UnifiedTokenizer
├── TokenizationConfig      # Configuration management
├── TokenizationStats       # Statistics tracking
├── CheckpointManager       # Checkpoint/resume logic
├── MonitoringMixin         # Progress monitoring
└── Processing Modes
    ├── Batch Mode          # High performance
    ├── Streaming Mode      # Memory safe
    └── Hybrid Mode         # Parallel processing
```

### Processing Modes

#### Batch Mode (Default)
- Processes documents in configurable batches
- Optimized for speed: 4,300+ docs/sec
- Best for systems with adequate memory

#### Streaming Mode
- Processes documents one at a time
- Memory-safe: ~600 docs/sec
- Ideal for very large files or limited memory

#### Hybrid Mode
- Combines batch processing with parallel workers
- Scales linearly with CPU cores
- Best for production workloads

## Performance

### Benchmarks
Based on FinPile dataset processing (52GB, 200 files):

| Mode | Docs/sec | Memory Usage | Use Case |
|------|----------|--------------|----------|
| Batch | 4,300+ | 3-4 GB | Fast processing, adequate memory |
| Streaming | 600 | 1-2 GB | Large files, limited memory |
| Hybrid (4 workers) | 15,000+ | 4-8 GB | Production, multi-core systems |

### 7.2x Performance Improvement
The unified tokenizer achieved a 7.2x speedup over the original implementation through:
- Batch tokenization with HuggingFace fast tokenizers (Rust backend)
- Efficient memory management with chunked processing
- Parallel processing support
- Optimized I/O with Arrow/Parquet format

## Usage Guide

### Command Line Options

```bash
python data_decide/scripts/tokenize.py INPUT OUTPUT [OPTIONS]

Required Arguments:
  INPUT               Input file or directory
  OUTPUT              Output directory

Processing Options:
  --mode              MODE     Processing mode: batch, streaming, hybrid (default: hybrid)
  --batch-size        SIZE     Batch size for tokenization (default: 200)
  --chunk-size        SIZE     Documents per chunk (default: 10000)
  --num-workers       N        Number of parallel workers (default: 1)

Tokenizer Options:
  --tokenizer         NAME     Tokenizer model name (default: EleutherAI/gpt-neox-20b)
  --max-seq-length    LENGTH   Maximum sequence length (default: 2048)
  --no-append-eos              Don't append EOS token

Checkpoint Options:
  --no-checkpoint              Disable checkpointing
  --checkpoint-interval N      Checkpoint every N files (default: 5)
  --no-resume                  Don't resume from checkpoint
  --checkpoint-dir    DIR      Checkpoint directory

Output Options:
  --output-format     FORMAT   Output format: arrow, parquet (default: arrow)
  --compression       TYPE     Compression: gzip, snappy, None

Monitoring Options:
  --no-rich-ui                 Disable Rich UI
  --report-interval   N        Progress report interval (default: 1000)
```

### Production Example

For production tokenization of large datasets:

```bash
# Launch with optimal settings
python data_decide/scripts/tokenize.py \
    /mnt/data/finpile \
    data/tokenized/finpile \
    --mode hybrid \
    --num-workers 8 \
    --output-format parquet \
    --compression snappy \
    --checkpoint-interval 10
```

### Monitoring Progress

In a separate terminal:
```bash
python data_decide/scripts/monitor_progress.py \
    data/tokenized/finpile \
    --checkpoint-dir tokenization_checkpoints \
    --continuous
```

## Advanced Features

### Checkpoint and Resume
The tokenizer automatically saves progress every N files (configurable). If interrupted:
- Progress is saved atomically to prevent corruption
- Resume automatically detects and skips completed files
- No data loss or duplication

### Memory Management
- Configurable memory limits with `--max-memory-gb`
- Automatic garbage collection after each save
- Memory usage monitoring in real-time

### Trust Remote Code
For tokenizers requiring custom code (e.g., OLMo):
```python
# Automatically handled with fallback
try:
    tokenizer = AutoTokenizer.from_pretrained(name)
except ValueError as e:
    if "trust_remote_code" in str(e):
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
```

### Custom Input Formats
The tokenizer auto-detects format or can be specified:
- `.json` - Single JSON file
- `.jsonl` - JSON Lines format
- `.json.gz` - Compressed JSON
- `.jsonl.gz` - Compressed JSON Lines
- `.txt` - Plain text files

## Troubleshooting

### Common Issues

#### Pickle Error with Multiprocessing
**Error**: `cannot pickle '_thread.RLock' object`
**Solution**: Use single worker mode: `--num-workers 1`

#### OOM (Out of Memory) Errors
**Solutions**:
1. Switch to streaming mode: `--mode streaming`
2. Reduce batch size: `--batch-size 100`
3. Set memory limit: `--max-memory-gb 4`

#### Tokenizer Trust Issues
**Error**: `trust_remote_code` required
**Solution**: The tokenizer automatically handles this, no action needed

### Performance Optimization

For best performance:
1. Use hybrid mode with multiple workers
2. Output to fast SSD storage
3. Use parquet format with snappy compression
4. Increase batch size if memory allows

### Validation

Always validate output:
```bash
# Check output statistics
python data_decide/scripts/monitor_progress.py output/dir

# Verify sequence counts match expected
ls -la output/dir/*.parquet | wc -l
```

## Migration from Legacy Scripts

### From Old Scripts
```bash
# Old: tokenize_finpile_hybrid.py
python tokenize_finpile_hybrid.py input output

# New: tokenize.py
python data_decide/scripts/tokenize.py input output --mode hybrid
```

### Feature Mapping
- `tokenize_finpile_streaming.py` → `tokenize.py --mode streaming`
- `tokenize_finpile_hybrid.py` → `tokenize.py --mode hybrid`
- `count_tokens.py --sample` → `count_tokens_unified.py`
- `count_exact_tokens.py` → `count_tokens_unified.py --exact`

## Summary

The unified tokenization system provides:
- **Performance**: Up to 15,000+ docs/sec with parallel processing
- **Reliability**: Checkpoint/resume for fault tolerance
- **Flexibility**: Multiple modes for different use cases
- **Monitoring**: Real-time progress and statistics
- **Compatibility**: Supports all major tokenizers and formats

For questions or issues, refer to the [main README](README.md) or check the [training guide](TRAINING_GUIDE.md).
