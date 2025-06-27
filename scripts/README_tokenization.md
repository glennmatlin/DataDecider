# DataDecider Tokenization Scripts

## Production Scripts

### 1. `tokenize_finpile_hybrid.py` (Recommended)
**Purpose**: High-performance tokenization with balanced memory usage
**Performance**: ~4,300 docs/sec (7.2x faster than streaming)
**Memory**: ~3 GB per worker

```bash
# Single worker (safe for limited memory)
python scripts/tokenize_finpile_hybrid.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --num-workers 1

# Multiple workers (62GB+ RAM recommended)
python scripts/tokenize_finpile_hybrid.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --num-workers 2 \
    --batch-size 200
```

**Key Features**:
- Batch tokenization for speed
- Memory-efficient chunking
- Parallel processing support
- Parquet output format
- Checkpoint/resume capability

### 2. `tokenize_finpile_streaming.py` (Fallback)
**Purpose**: Memory-safe tokenization for constrained environments
**Performance**: ~600 docs/sec
**Memory**: 2-3 GB stable

```bash
python scripts/tokenize_finpile_streaming.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --sequences-per-chunk 10000
```

**Use When**:
- Memory is critically limited
- Maximum reliability needed
- Running on shared resources

### 3. `tokenize_datasets.py` (General Purpose)
**Purpose**: Tokenize various dataset formats for training
**Supports**: JSON, JSONL, GZ compressed files

```bash
# Tokenize with specific parameters
python scripts/tokenize_datasets.py \
    --input-path /path/to/data.json \
    --output-dir /path/to/output \
    --tokenizer EleutherAI/gpt-neox-20b \
    --max-length 2048 \
    --validation-split 0.1
```

## Utility Scripts

### `check_finpile_progress.py`
Monitor tokenization progress and estimate completion time:
```bash
python scripts/check_finpile_progress.py \
    --output-dir /path/to/tokenized/output
```

## Performance Comparison

| Script | Speed | Memory | Use Case |
|--------|-------|--------|----------|
| hybrid | 4,300 docs/s | 3 GB/worker | Production (recommended) |
| streaming | 600 docs/s | 2-3 GB | Memory-constrained |
| datasets | Variable | Variable | General datasets |

## Archived Scripts

Experimental and deprecated scripts are preserved in `scripts/archived_tokenizers/`:
- `failed/` - Scripts with critical issues
- `experimental/` - Test implementations
- `benchmarks/` - Performance testing tools

## Best Practices

1. **For FinPile tokenization**: Use `tokenize_finpile_hybrid.py`
2. **Memory calculation**: ~3 GB per worker + 2 GB base
3. **Optimal workers**: `min(CPU_cores/4, available_RAM_GB/4)`
4. **Monitor progress**: Use separate terminal with `check_finpile_progress.py`
5. **Resume on failure**: Scripts automatically checkpoint progress

## Troubleshooting

**Out of Memory**:
- Reduce `--num-workers`
- Decrease `--batch-size`
- Use streaming tokenizer as fallback

**Slow Performance**:
- Increase `--batch-size` (if memory allows)
- Add more workers
- Ensure using fast tokenizer (`use_fast=True`)

**Permission Errors**:
- Check output directory permissions
- Ensure no conflicting temp files
- Use absolute paths
