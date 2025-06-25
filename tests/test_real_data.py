"""Test OLMo data curation with real arXiv data."""

import os
import sys
import gzip
import json
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.data_curation import DataDecideCurator
from olmo.data.preprocessing import OLMoDataPreprocessor
from transformers import AutoTokenizer


def test_arxiv_data_loading():
    """Test loading real arXiv data."""
    data_path = Path(__file__).parent / "test_data"
    
    # Create curator
    curator = DataDecideCurator(
        data_path=str(data_path),
        tokenizer_name="gpt2",  # Use GPT-2 tokenizer as placeholder
        proxy_model_size="150M"
    )
    
    # Load data
    print("Loading arXiv data...")
    data = curator.load_json_data()
    print(f"Loaded {len(data)} documents")
    
    # Compute statistics
    print("\nComputing data statistics...")
    stats = curator.compute_data_statistics(data)
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Create proxy experiments
    print("\nCreating proxy experiments...")
    proxy_datasets = curator.create_proxy_experiments(data, num_experiments=3)
    print(f"Created {len(proxy_datasets)} proxy datasets")
    
    # Evaluate proxy datasets
    print("\nEvaluating proxy datasets...")
    scores = curator.evaluate_proxy_datasets(proxy_datasets)
    for dataset_name, metrics in scores.items():
        print(f"\n{dataset_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Select best recipe
    best_recipe = curator.select_best_data_recipe(scores)
    print(f"\nBest data recipe: {best_recipe}")
    
    return data, stats, scores


def test_preprocessing_pipeline():
    """Test preprocessing pipeline on arXiv data."""
    data_path = Path(__file__).parent / "test_data"
    
    # Load a few documents
    documents = []
    arxiv_file = data_path / "arxiv_sample.json.gz"
    
    with gzip.open(arxiv_file, 'rt') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Just process 5 documents
                break
            documents.append(json.loads(line))
    
    print(f"\nTesting preprocessing on {len(documents)} documents...")
    
    # Initialize tokenizer and preprocessor
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    
    preprocessor = OLMoDataPreprocessor(
        tokenizer=tokenizer,
        max_length=512,
        concatenate_documents=True,
        add_eos_token=True
    )
    
    # Create a simple dataset format
    from datasets import Dataset
    dataset = Dataset.from_list([{'text': doc['text'][:2000]} for doc in documents])
    
    # Preprocess
    print("Preprocessing documents...")
    processed = preprocessor.create_training_dataset(dataset, num_proc=1)
    
    print(f"Processed dataset size: {len(processed)}")
    print(f"First example shape: {processed[0]['input_ids'].shape}")
    
    # Show sample
    sample_ids = processed[0]['input_ids'][:50]
    sample_text = tokenizer.decode(sample_ids)
    print(f"\nSample decoded text (first 50 tokens):\n{sample_text}")
    
    return processed


def analyze_arxiv_content():
    """Analyze content characteristics of arXiv data."""
    data_path = Path(__file__).parent / "test_data"
    arxiv_file = data_path / "arxiv_sample.json.gz"
    
    print("\n=== ArXiv Data Analysis ===")
    
    # Collect statistics
    doc_lengths = []
    urls = []
    text_samples = []
    
    with gzip.open(arxiv_file, 'rt') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            doc_lengths.append(len(doc['text']))
            urls.append(doc['url'])
            if i < 3:  # Keep first 3 samples
                text_samples.append(doc['text'][:500] + "...")
    
    print(f"\nTotal documents: {len(doc_lengths)}")
    print(f"Average document length: {sum(doc_lengths) / len(doc_lengths):.0f} characters")
    print(f"Min length: {min(doc_lengths)} characters")
    print(f"Max length: {max(doc_lengths)} characters")
    
    print("\nSample URLs:")
    for url in urls[:5]:
        print(f"  - {url}")
    
    print("\nSample texts:")
    for i, text in enumerate(text_samples):
        print(f"\n--- Document {i+1} ---")
        print(text)


if __name__ == "__main__":
    # Check if test data exists
    test_data_path = Path(__file__).parent / "test_data" / "arxiv_sample.json.gz"
    if not test_data_path.exists():
        print(f"Error: Test data not found at {test_data_path}")
        print("Please ensure arxiv_sample.json.gz is in the test_data directory")
        sys.exit(1)
    
    # Run tests
    print("Testing OLMo Data Curation with Real ArXiv Data")
    print("=" * 50)
    
    # Analyze content
    analyze_arxiv_content()
    
    # Test data loading and curation
    data, stats, scores = test_arxiv_data_loading()
    
    # Test preprocessing
    processed_dataset = test_preprocessing_pipeline()
    
    print("\n✓ All tests completed successfully!")