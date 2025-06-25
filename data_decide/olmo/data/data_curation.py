# src/data/data_curation.py
import json
import os
from typing import List, Dict, Any
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import hashlib


class DataDecideCurator:
    """
    Implements DataDecide methodology for data curation and selection.
    Uses small-scale experiments to predict optimal data composition.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "allenai/OLMo-7B",
        proxy_model_size: str = "150M",
        num_proxy_steps: int = 1000,
        eval_metrics: List[str] = ["perplexity", "diversity", "quality"],
    ):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.proxy_model_size = proxy_model_size
        self.num_proxy_steps = num_proxy_steps
        self.eval_metrics = eval_metrics

    def load_json_data(self, file_pattern: str = "*.json") -> List[Dict[str, Any]]:
        """Load JSON data from folder following the specified format."""
        # Check if this is a pre-tokenized HuggingFace dataset
        dataset_info_path = os.path.join(self.data_path, "dataset_dict.json")
        if os.path.exists(dataset_info_path):
            return self._load_huggingface_dataset()
        
        # Otherwise, load JSON/JSONL files
        data_files = []
        for file in os.listdir(self.data_path):
            if file.endswith(".json") or file.endswith(".json.gz"):
                data_files.append(os.path.join(self.data_path, file))

        all_data = []
        for file_path in tqdm(data_files, desc="Loading JSON files"):
            if file_path.endswith(".gz"):
                import gzip

                with gzip.open(file_path, "rt") as f:
                    for line in f:
                        all_data.append(json.loads(line))
            else:
                with open(file_path, "r") as f:
                    for line in f:
                        all_data.append(json.loads(line))

        return all_data
    
    def _load_huggingface_dataset(self) -> List[Dict[str, Any]]:
        """Load pre-tokenized HuggingFace dataset."""
        from datasets import DatasetDict
        
        # Load the dataset
        dataset_dict = DatasetDict.load_from_disk(self.data_path)
        
        # For DataDecide, we'll use the training split
        train_dataset = dataset_dict['train']
        
        # Convert to list of dicts format expected by rest of pipeline
        # Since this is pre-tokenized, we'll create synthetic "text" from tokens
        all_data = []
        for i in range(min(10000, len(train_dataset))):  # Sample for proxy experiments
            item = train_dataset[i]
            # Create a unique identifier
            doc = {
                "text": f"<pre-tokenized-doc-{i}>",  # Placeholder since already tokenized
                "uuid": f"doc-{i}",
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"]
            }
            all_data.append(doc)
        
        return all_data

    def compute_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute statistics for data quality assessment."""
        stats = {
            "total_documents": len(data),
            "avg_length": 0,
            "vocabulary_size": 0,
            "deduplication_rate": 0,
            "quality_score": 0,
        }

        # Length statistics
        lengths = [len(doc["text"]) for doc in data]
        stats["avg_length"] = np.mean(lengths)
        stats["std_length"] = np.std(lengths)

        # Vocabulary analysis
        vocab = set()
        for doc in data[:10000]:  # Sample for efficiency
            vocab.update(doc["text"].split())
        stats["vocabulary_size"] = len(vocab)

        # Deduplication analysis
        hashes = set()
        duplicates = 0
        for doc in data:
            doc_hash = hashlib.md5(doc["text"].encode()).hexdigest()
            if doc_hash in hashes:
                duplicates += 1
            hashes.add(doc_hash)
        stats["deduplication_rate"] = duplicates / len(data)

        return stats

    def create_proxy_experiments(
        self, data: List[Dict[str, Any]], num_experiments: int = 5
    ) -> List[Dataset]:
        """Create small-scale proxy datasets for DataDecide experiments."""
        # For pre-tokenized data, we'll create different data mixtures
        # This follows the DataDecide paper's approach of testing different data recipes
        
        # Check if data is pre-tokenized
        is_pretokenized = "input_ids" in data[0] if data else False
        
        if is_pretokenized:
            # For pre-tokenized data, load the full dataset and create different subsets
            from datasets import DatasetDict
            dataset_dict = DatasetDict.load_from_disk(self.data_path)
            train_dataset = dataset_dict['train']
            
            proxy_datasets = []
            dataset_size = len(train_dataset)
            
            # Create different data recipes by sampling different portions
            sampling_strategies = [
                ("random_10k", 10000),
                ("random_20k", 20000),
                ("random_40k", 40000),
                ("stratified_20k", 20000),  # Will implement stratified sampling
                ("full_subset", min(50000, dataset_size))
            ]
            
            for i, (strategy_name, sample_size) in enumerate(sampling_strategies[:num_experiments]):
                if sample_size > dataset_size:
                    sample_size = dataset_size
                
                # Random sampling
                indices = np.random.choice(dataset_size, size=sample_size, replace=False)
                subset = train_dataset.select(indices)
                
                proxy_datasets.append(subset)
            
            return proxy_datasets
        
        # Original implementation for non-tokenized data
        proxy_datasets = []

        for i in range(num_experiments):
            # Different sampling strategies
            if i == 0:
                # Random sampling
                sampled = np.random.choice(
                    data, size=min(10000, len(data)), replace=False
                )
            elif i == 1:
                # Length-based sampling (prefer medium length)
                lengths = np.array([len(doc["text"]) for doc in data])
                weights = np.exp(
                    -((lengths - np.median(lengths)) ** 2) / (2 * np.std(lengths) ** 2)
                )
                weights /= weights.sum()
                sampled = np.random.choice(
                    data, size=min(10000, len(data)), replace=False, p=weights
                )
            elif i == 2:
                # Quality-based sampling (using simple heuristics)
                sampled = [doc for doc in data if self._quality_filter(doc)][:10000]
            elif i == 3:
                # Diversity-based sampling
                sampled = self._diversity_sample(data, 10000)
            else:
                # Mixed strategy
                sampled = self._mixed_strategy_sample(data, 10000)

            # Convert to HuggingFace Dataset
            dataset = Dataset.from_list(
                [{"text": doc["text"], "uuid": doc["uuid"]} for doc in sampled]
            )
            proxy_datasets.append(dataset)

        return proxy_datasets

    def _quality_filter(self, doc: Dict[str, Any]) -> bool:
        """Simple quality filter based on heuristics."""
        text = doc["text"]
        # Basic quality checks
        if len(text) < 100 or len(text) > 50000:
            return False
        if text.count("\n") / len(text) > 0.3:  # Too many newlines
            return False
        if len(set(text)) / len(text) < 0.01:  # Low character diversity
            return False
        return True

    def _diversity_sample(
        self, data: List[Dict[str, Any]], n_samples: int
    ) -> List[Dict[str, Any]]:
        """Sample for maximum diversity using clustering."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans

        # Sample subset for efficiency
        subset = np.random.choice(data, size=min(50000, len(data)), replace=False)
        texts = [doc["text"][:1000] for doc in subset]  # Use first 1000 chars

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        features = vectorizer.fit_transform(texts)

        # Clustering
        n_clusters = min(n_samples, len(subset))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Sample one from each cluster
        sampled = []
        for i in range(n_clusters):
            cluster_docs = [subset[j] for j in range(len(subset)) if labels[j] == i]
            if cluster_docs:
                sampled.append(np.random.choice(cluster_docs))

        return sampled

    def _mixed_strategy_sample(
        self, data: List[Dict[str, Any]], n_samples: int
    ) -> List[Dict[str, Any]]:
        """Combine multiple sampling strategies."""
        strategies = [
            lambda d, n: np.random.choice(d, size=n, replace=False),
            lambda d, n: [doc for doc in d if self._quality_filter(doc)][:n],
            lambda d, n: self._diversity_sample(d, n),
        ]

        samples_per_strategy = n_samples // len(strategies)
        sampled = []

        for strategy in strategies:
            sampled.extend(strategy(data, samples_per_strategy))

        return sampled[:n_samples]

    def evaluate_proxy_datasets(
        self, proxy_datasets: List[Dataset]
    ) -> Dict[str, float]:
        """Evaluate proxy datasets to predict performance."""
        scores = {}

        for i, dataset in enumerate(proxy_datasets):
            # Compute various metrics
            perplexity = self._compute_proxy_perplexity(dataset)
            diversity = self._compute_diversity_score(dataset)
            quality = self._compute_quality_score(dataset)

            scores[f"dataset_{i}"] = {
                "perplexity": perplexity,
                "diversity": diversity,
                "quality": quality,
                "combined_score": (1 / perplexity) * diversity * quality,
            }

        return scores

    def _compute_proxy_perplexity(self, dataset: Dataset) -> float:
        """Compute perplexity using a small proxy model.
        
        This follows the DataDecide methodology where we use small-scale
        experiments to predict performance at larger scales.
        """
        # For pre-tokenized data, we can estimate perplexity based on token statistics
        if "input_ids" in dataset.column_names:
            # Calculate entropy-based metric as proxy for perplexity
            all_tokens = []
            for item in dataset.select(range(min(1000, len(dataset)))):
                all_tokens.extend(item["input_ids"])
            
            # Calculate token frequency distribution
            token_counts = {}
            for token in all_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            # Calculate entropy
            total_tokens = len(all_tokens)
            entropy = 0
            for count in token_counts.values():
                prob = count / total_tokens
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            
            # Convert entropy to perplexity-like score (2^entropy)
            # Normalize to typical perplexity range
            perplexity = min(200, max(50, 2 ** (entropy / 4)))
            return perplexity
        
        # For text data, use simple heuristics
        return 100.0  # Default middle-range perplexity

    def _compute_diversity_score(self, dataset: Dataset) -> float:
        """Compute diversity score of dataset.
        
        Following DataDecide, we measure how diverse the data is
        to predict if it will lead to better model performance.
        """
        if "input_ids" in dataset.column_names:
            # For tokenized data, compute type-token ratio
            all_tokens = []
            sample_size = min(5000, len(dataset))
            for i in range(sample_size):
                all_tokens.extend(dataset[i]["input_ids"])
            
            unique_tokens = len(set(all_tokens))
            total_tokens = len(all_tokens)
            
            # Type-token ratio (normalized)
            ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
            
            # Calculate n-gram diversity (bigrams)
            bigrams = set()
            for i in range(len(all_tokens) - 1):
                bigrams.add((all_tokens[i], all_tokens[i+1]))
            
            bigram_diversity = len(bigrams) / (len(all_tokens) - 1) if len(all_tokens) > 1 else 0
            
            # Combined diversity score
            diversity = 0.7 * ttr + 0.3 * bigram_diversity
            return min(1.0, diversity * 2)  # Scale to 0-1 range
        
        # For text data
        texts = [item["text"] for item in dataset.select(range(min(1000, len(dataset))))]
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())
        
        return min(1.0, len(vocab) / 10000)  # Normalize by expected vocab size

    def _compute_quality_score(self, dataset: Dataset) -> float:
        """Compute quality score based on various heuristics.
        
        For DataDecide, quality is a key factor in predicting
        which data will produce better models.
        """
        if "input_ids" in dataset.column_names:
            # For pre-tokenized arXiv data, we assume high quality
            # We can check for things like sequence length distribution
            lengths = []
            sample_size = min(1000, len(dataset))
            for i in range(sample_size):
                # Count non-padding tokens
                tokens = dataset[i]["input_ids"]
                if isinstance(tokens, list):
                    length = len([t for t in tokens if t != 0])  # Assuming 0 is padding
                else:
                    length = len(tokens)
                lengths.append(length)
            
            # Good quality data has consistent, reasonable length sequences
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            # Quality based on length consistency and reasonable average
            if 500 < avg_length < 1800 and std_length < 500:
                quality = 0.9
            elif 100 < avg_length < 2000:
                quality = 0.8
            else:
                quality = 0.7
            
            return quality
        
        # For text data, use original heuristics
        return 0.85  # Default high quality for arXiv

    def select_best_data_recipe(self, scores: Dict[str, float]) -> str:
        """Select best data recipe based on proxy experiments."""
        best_dataset = max(scores.items(), key=lambda x: x[1]["combined_score"])
        return best_dataset[0]
