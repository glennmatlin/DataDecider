"""Unit tests for data curation using real arXiv data."""

import unittest
import tempfile
import json
import gzip
import os
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.data_curation import DataDecideCurator
from datasets import Dataset


class TestDataCurationArxiv(unittest.TestCase):
    """Test data curation with real arXiv data format."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.arxiv_file = cls.test_data_dir / "arxiv_sample.json.gz"
        
        # Load sample documents
        cls.sample_docs = []
        if cls.arxiv_file.exists():
            with gzip.open(cls.arxiv_file, 'rt') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Load first 10 documents
                        break
                    cls.sample_docs.append(json.loads(line))
    
    def setUp(self):
        """Set up for each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Write sample data to temp directory
        temp_file = os.path.join(self.temp_dir, "test_arxiv.json")
        with open(temp_file, 'w') as f:
            for doc in self.sample_docs[:5]:  # Use first 5 docs
                f.write(json.dumps(doc) + '\n')
    
    def tearDown(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_arxiv_json_data(self):
        """Test loading arXiv JSON data."""
        curator = DataDecideCurator(self.temp_dir)
        data = curator.load_json_data()
        
        self.assertEqual(len(data), 5)
        
        # Check data structure
        for doc in data:
            self.assertIn('url', doc)
            self.assertIn('shard', doc)
            self.assertIn('text', doc)
            self.assertTrue(doc['url'].startswith('https://arxiv.org/'))
    
    def test_arxiv_data_statistics(self):
        """Test computing statistics on arXiv data."""
        curator = DataDecideCurator(self.temp_dir)
        data = curator.load_json_data()
        stats = curator.compute_data_statistics(data)
        
        # Check computed statistics
        self.assertEqual(stats['total_documents'], 5)
        self.assertGreater(stats['avg_length'], 0)
        self.assertGreater(stats['vocabulary_size'], 0)
        self.assertGreaterEqual(stats['deduplication_rate'], 0)
        self.assertLessEqual(stats['deduplication_rate'], 1)
    
    def test_quality_filter_arxiv(self):
        """Test quality filtering on arXiv documents."""
        curator = DataDecideCurator(self.temp_dir)
        
        # Test with real arXiv document
        if self.sample_docs:
            doc = self.sample_docs[0]
            result = curator._quality_filter(doc)
            self.assertIsInstance(result, bool)
            
            # Most arXiv papers should pass quality filter
            if len(doc['text']) > 100 and len(doc['text']) < 50000:
                self.assertTrue(result)
    
    def test_create_proxy_experiments_arxiv(self):
        """Test creating proxy experiments with arXiv data."""
        curator = DataDecideCurator(self.temp_dir)
        data = curator.load_json_data()
        
        # Create proxy datasets
        proxy_datasets = curator.create_proxy_experiments(data, num_experiments=3)
        
        self.assertEqual(len(proxy_datasets), 3)
        
        for dataset in proxy_datasets:
            self.assertIsInstance(dataset, Dataset)
            self.assertGreater(len(dataset), 0)
            
            # Check dataset format
            if len(dataset) > 0:
                self.assertIn('text', dataset[0])
                self.assertIn('uuid', dataset[0])
    
    def test_arxiv_document_structure(self):
        """Test that arXiv documents have expected structure."""
        if not self.sample_docs:
            self.skipTest("No sample documents loaded")
        
        doc = self.sample_docs[0]
        
        # Check required fields
        self.assertIn('url', doc)
        self.assertIn('shard', doc)
        self.assertIn('text', doc)
        
        # Check URL format
        self.assertTrue(doc['url'].startswith('https://arxiv.org/abs/'))
        
        # Check text contains LaTeX (common in arXiv)
        self.assertIn('\\', doc['text'])  # LaTeX commands
        
        # Check shard naming
        self.assertTrue(doc['shard'].endswith('.json.gz'))
    
    def test_diversity_sampling_arxiv(self):
        """Test diversity sampling with scientific text."""
        if len(self.sample_docs) < 10:
            self.skipTest("Not enough sample documents for diversity test")
        
        curator = DataDecideCurator(self.temp_dir)
        
        # Use more documents for diversity sampling
        sampled = curator._diversity_sample(self.sample_docs[:10], n_samples=3)
        
        self.assertLessEqual(len(sampled), 3)
        self.assertGreater(len(sampled), 0)
        
        # Check that sampled documents are from original set
        for doc in sampled:
            self.assertIn(doc, self.sample_docs[:10])
    
    def test_mixed_strategy_sampling(self):
        """Test mixed strategy sampling."""
        curator = DataDecideCurator(self.temp_dir)
        data = curator.load_json_data()
        
        if len(data) < 3:
            self.skipTest("Not enough data for mixed strategy test")
        
        sampled = curator._mixed_strategy_sample(data, n_samples=3)
        
        self.assertLessEqual(len(sampled), 3)
        self.assertGreater(len(sampled), 0)


class TestArxivDataCharacteristics(unittest.TestCase):
    """Test characteristics specific to arXiv data."""
    
    @classmethod
    def setUpClass(cls):
        """Load sample arXiv data."""
        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.arxiv_file = cls.test_data_dir / "arxiv_sample.json.gz"
        
        cls.documents = []
        if cls.arxiv_file.exists():
            with gzip.open(cls.arxiv_file, 'rt') as f:
                for i, line in enumerate(f):
                    if i >= 20:  # Load more documents for analysis
                        break
                    cls.documents.append(json.loads(line))
    
    def test_latex_content(self):
        """Test that arXiv documents contain LaTeX."""
        if not self.documents:
            self.skipTest("No documents loaded")
        
        latex_docs = 0
        for doc in self.documents:
            if '\\section' in doc['text'] or '\\begin' in doc['text']:
                latex_docs += 1
        
        # Most arXiv papers should contain LaTeX
        self.assertGreater(latex_docs / len(self.documents), 0.8)
    
    def test_document_lengths(self):
        """Test document length distribution."""
        if not self.documents:
            self.skipTest("No documents loaded")
        
        lengths = [len(doc['text']) for doc in self.documents]
        
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        # arXiv papers are typically substantial
        self.assertGreater(avg_length, 10000)  # At least 10k characters on average
        self.assertGreater(min_length, 1000)   # Even short papers > 1k chars
        self.assertLess(max_length, 1000000)   # Reasonable upper bound
    
    def test_scientific_vocabulary(self):
        """Test for scientific vocabulary in documents."""
        if not self.documents:
            self.skipTest("No documents loaded")
        
        scientific_terms = [
            'theorem', 'proof', 'lemma', 'equation', 'algorithm',
            'experiment', 'hypothesis', 'result', 'figure', 'table'
        ]
        
        docs_with_terms = 0
        for doc in self.documents[:10]:  # Check first 10
            text_lower = doc['text'].lower()
            if any(term in text_lower for term in scientific_terms):
                docs_with_terms += 1
        
        # Most documents should contain scientific terms
        self.assertGreater(docs_with_terms / min(10, len(self.documents)), 0.7)


if __name__ == '__main__':
    # Check if test data exists
    test_data_path = Path(__file__).parent / "test_data" / "arxiv_sample.json.gz"
    if not test_data_path.exists():
        print(f"Warning: Test data not found at {test_data_path}")
        print("Some tests will be skipped")
    
    unittest.main()