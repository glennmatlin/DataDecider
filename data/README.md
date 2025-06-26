# Data Directory

This directory contains the datasets used for training OLMo models with the DataDecide methodology.

## Directory Structure

```
data/
├── raw/              # Raw data files (JSON, JSONL, etc.)
│   └── arxiv_sample.json.gz  # Small sample for testing
├── processed/        # Tokenized datasets in HuggingFace format
└── README.md         # This file
```

## Obtaining Full Datasets

The full datasets are not included in this repository due to their large size (9.6GB+). To obtain them:

### 1. ArXiv Dataset (400M tokens)

The ArXiv dataset used for OLMo 4M training contains approximately 400M tokens from scientific papers.

**Option A: Download from Dolma**
```bash
# Download specific arXiv shards from Dolma v1.7
# Visit: https://huggingface.co/datasets/allenai/dolma

# Example shards used:
wget https://huggingface.co/datasets/allenai/dolma/resolve/main/data/arxiv/arxiv-0012.json.gz
wget https://huggingface.co/datasets/allenai/dolma/resolve/main/data/arxiv/arxiv-0017.json.gz
# ... (see full list in configs/data_configs/arxiv_shards.txt)
```

**Option B: Build from scratch**
```bash
# Use the data preparation script
python -m data_decide.scripts.prepare_training_data \
    --input-dir ./raw_arxiv_papers \
    --output-dir ./data/processed/arxiv_400M \
    --tokenizer EleutherAI/gpt-neox-20b \
    --max-tokens 400000000
```

### 2. Custom Datasets

To prepare your own datasets:

```bash
# Tokenize and prepare custom data
python -m data_decide.scripts.build_4m_dataset \
    --input-files ./your_data/*.jsonl \
    --output-dir ./data/processed/custom_dataset \
    --tokenizer EleutherAI/gpt-neox-20b \
    --train-split 0.95 \
    --max-length 2048
```

## Data Format

### Raw Data Format (JSONL)
Each line should be a JSON object with at least a "text" field:
```json
{"text": "Your document text here...", "metadata": {...}}
```

### Processed Data Format
Processed datasets are saved as HuggingFace datasets with:
- `input_ids`: Tokenized text
- `attention_mask`: Attention mask for padding
- `labels`: Same as input_ids for causal language modeling

## Storage Requirements

- Sample dataset: ~1MB (included)
- Full ArXiv 400M dataset: ~4.9GB
- Full Dolma v1.7: Multiple TB (not needed for basic experiments)

## Using Pre-trained Datasets

If you have access to pre-tokenized datasets, place them in:
```
data/processed/[dataset_name]/
├── dataset_dict.json
├── train/
│   ├── data-*.arrow
│   ├── dataset_info.json
│   └── state.json
└── validation/
    ├── data-*.arrow
    ├── dataset_info.json
    └── state.json
```

## Data Quality

The DataDecide methodology relies on high-quality data. Ensure your datasets:
- Are properly cleaned (no HTML, minimal noise)
- Have consistent formatting
- Are deduplicated
- Match your target domain

## Citation

If using the Dolma dataset:
```bibtex
@article{dolma,
  title={Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research},
  author={Soldaini, Luca and others},
  journal={arXiv preprint arXiv:2402.00159},
  year={2024}
}
```
