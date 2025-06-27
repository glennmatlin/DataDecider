# Tokenization Performance Improvements

## Executive Summary

Through implementing batch mapping and concatenated tokenization techniques suggested by the community, we achieved **7.2x performance improvement** while maintaining reasonable memory usage.

## Performance Comparison

| Method | Speed (docs/sec) | Memory Usage | Reliability | Production Ready |
|--------|------------------|--------------|-------------|------------------|
| Original (failed) | ~1,146 | 50+ GB (OOM) | ❌ Failed | No |
| Streaming (current) | ~600 | 2-3 GB | ✅ Stable | Yes |
| Ultra-fast (parallel) | ~10,000* | 48+ GB (OOM) | ❌ Failed | No |
| **Hybrid (new)** | **4,331** | **3.1 GB** | **✅ Stable** | **Yes** |

*Theoretical based on single-worker performance

## Key Improvements Implemented

### 1. Batch Tokenization
Instead of tokenizing documents one-by-one:
```python
# Old approach (slow)
for doc in documents:
    tokens = tokenizer(doc['text'])
```

We now process in batches:
```python
# New approach (fast)
batch_tokens = tokenizer(texts, batched=True)
```

### 2. Optimized Chunking
- Process 10,000 documents at a time
- Tokenize in batches of 200
- Save every 25,000 sequences
- Memory peaks at ~3 GB per worker

### 3. Efficient Storage
- Switched from Arrow to Parquet format
- Better compression and streaming support
- Supports incremental writes

## Technical Details

### What is Batch Mapping?
HuggingFace's batch mapping allows the tokenizer to process multiple texts simultaneously. The Rust-based fast tokenizers are optimized for this pattern, providing significant speedups.

### The "Huge Token String" Approach
The community suggestion about concatenating texts refers to:
1. Joining multiple documents with delimiter tokens
2. Tokenizing once (very efficient)
3. Splitting the result into sequences

We tested this but found batch tokenization more effective for our use case.

## Implementation

### Hybrid Tokenizer Features
- **Batch Processing**: 200 documents per batch
- **Memory Control**: Saves every 25k sequences
- **Parallel Support**: Can use multiple workers
- **Resume Capability**: Via checkpoint files
- **Format**: Parquet for efficient storage

### Usage
```bash
# Single worker (recommended for memory constraints)
python scripts/tokenize_finpile_hybrid.py \
    --input-dir /mnt/z/FinPile/0fp-100dolma \
    --output-dir /mnt/z/FinPile/tokenized/0fp-100dolma_hybrid \
    --num-workers 2 \
    --batch-size 200 \
    --chunk-size 10000

# Multiple workers (if sufficient memory)
python scripts/tokenize_finpile_hybrid.py \
    --input-dir /mnt/z/FinPile/0fp-100dolma \
    --output-dir /mnt/z/FinPile/tokenized/0fp-100dolma_hybrid \
    --num-workers 4 \
    --batch-size 200 \
    --chunk-size 10000
```

## Performance Projections

For the full FinPile dataset (200 files):
- **Streaming tokenizer**: ~36 hours (current)
- **Hybrid tokenizer (1 worker)**: ~5 hours
- **Hybrid tokenizer (2 workers)**: ~2.5 hours
- **Hybrid tokenizer (4 workers)**: ~1.3 hours

## Memory Management

The hybrid approach maintains stable memory usage through:
1. **Chunked Reading**: Process files in 10k document chunks
2. **Periodic Saves**: Write to disk every 25k sequences
3. **Garbage Collection**: Explicit cleanup after each save
4. **Controlled Parallelism**: Limit workers based on available RAM

## Recommendations

1. **Immediate**: Switch to hybrid tokenizer for 7x speedup
2. **Production Settings**:
   - 2 workers for 62GB RAM system
   - Monitor memory usage
   - Use checkpoint/resume for reliability
3. **Future Optimizations**:
   - Experiment with larger batch sizes
   - Test memory-mapped file reading
   - Consider GPU-accelerated tokenization

## Conclusion

The batch mapping approach suggested by the community proved highly effective. By combining batch tokenization with careful memory management, we achieved:
- **7.2x speedup** over current approach
- **Stable memory usage** (3.1 GB vs 50+ GB)
- **Production reliability** with checkpointing
- **Parallel scalability** when memory allows

This represents a significant improvement in our data processing pipeline, reducing tokenization time from 36 hours to potentially 2.5 hours.
