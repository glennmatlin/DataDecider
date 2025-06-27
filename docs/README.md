# DataDecider Scripts

This directory contains the production scripts for the DataDecider project.

## 🚀 Production Scripts

### **tokenize_unified.py** - Unified Tokenization System
The main production tokenizer combining all best features:
- **Performance**: Up to 4,300 docs/sec in batch mode
- **Modes**: batch (fast), streaming (memory-safe), hybrid (balanced)
- **Features**: Checkpoint/resume, parallel processing, multiple formats
- **Memory**: Configurable memory limits with automatic management

```bash
# Basic usage
python tokenize_unified.py /mnt/z/FinPile/datamix/lda/0fp-100dolma data/tokenized/finpile_lda

# High-performance mode (8 workers)
python tokenize_unified.py input/ output/ --mode batch --num-workers 8

# Memory-safe mode (4GB limit)
python tokenize_unified.py input/ output/ --mode streaming --max-memory-gb 4

# Resume interrupted job
python tokenize_unified.py input/ output/ --resume
```

### **monitor_progress.py** - Universal Progress Monitor
Works with all tokenization formats (unified, streaming, hybrid, legacy):
- Auto-detects tokenization format
- Shows real-time progress and statistics
- Monitors running processes and resource usage
- Rich UI or simple text output

```bash
# Monitor unified tokenizer
python monitor_progress.py data/tokenized/finpile_lda --checkpoint-dir tokenization_checkpoints

# Continuous monitoring with Rich UI
python monitor_progress.py data/tokenized/finpile_lda --continuous

# Simple text output
python monitor_progress.py data/tokenized/finpile_lda --no-rich
```

### **tokenize_datasets.py** - General Purpose Tokenizer
For non-FinPile datasets with different formats:
- Handles JSON, JSONL, plain text
- Creates train/validation splits
- Generates comprehensive metadata
- Different use case than FinPile-specific scripts

```bash
python tokenize_datasets.py configs/dataset_configs/arxiv_sample.yaml
```

## 📁 Archived Scripts

The following scripts have been archived in `archived_scripts/` as their functionality is now incorporated into the unified system:

- **tokenize_finpile_hybrid.py** - Original hybrid tokenizer (features merged into unified)
- **tokenize_finpile_streaming.py** - Streaming tokenizer (streaming mode in unified)
- **check_finpile_progress.py** - Basic progress checker (replaced by monitor_progress.py)

## 🔧 Other Utilities

- **build_data.py** - Dataset preparation and validation
- **count_tokens.py** - Token counting utility
- Various model-specific scripts

## 💡 Quick Start

For tokenizing FinPile data:

```bash
# 1. Start tokenization with unified tokenizer
python tokenize_unified.py \
    /mnt/z/FinPile/datamix/lda/0fp-100dolma \
    data/tokenized/finpile_lda \
    --mode hybrid \
    --num-workers 4

# 2. Monitor progress in another terminal
python monitor_progress.py \
    data/tokenized/finpile_lda \
    --continuous
```

## 🏗️ Architecture

The unified tokenization system consolidates the best features:

| Feature | Source | Implementation |
|---------|--------|----------------|
| Batch tokenization | hybrid.py | 4,300 docs/sec performance |
| Memory safety | streaming.py | Configurable memory limits |
| Checkpoint/resume | streaming.py | Atomic file operations |
| Parallel processing | hybrid.py | Multi-worker support |
| Rich monitoring | New | Live progress dashboard |
| Format flexibility | datasets.py | Multiple input/output formats |

This consolidation eliminated ~750 lines of duplicate code while maintaining all functionality.
