"""OLMo model evaluator for benchmarking."""

from typing import Dict, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from ..utils import get_logger

logger = get_logger(__name__)


class OLMoEvaluator:
    """Comprehensive evaluation suite for OLMo models."""

    BENCHMARK_TASKS = [
        "piqa",
        "hellaswag",
        "winogrande",
        "arc_easy",
        "arc_challenge",
        "boolq",
        "openbookqa",
        "mmlu",
        "truthfulqa",
        "gsm8k",
    ]

    def __init__(self, model, tokenizer, device: str = "cuda", batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def evaluate_all_benchmarks(self) -> Dict[str, float]:
        """Evaluate on all benchmark tasks."""
        results = {}

        for task in self.BENCHMARK_TASKS:
            logger.info(f"Evaluating on {task}...")
            score = self.evaluate_task(task)
            results[task] = score
            logger.info(f"{task}: {score:.4f}")

        # Calculate average
        results["average"] = np.mean(list(results.values()))

        return results

    def evaluate_task(self, task_name: str) -> float:
        """Evaluate on a specific task."""
        if task_name == "mmlu":
            return self._evaluate_mmlu()
        elif task_name == "gsm8k":
            return self._evaluate_gsm8k()
        elif task_name == "truthfulqa":
            return self._evaluate_truthfulqa()
        else:
            return self._evaluate_multiple_choice(task_name)

    def _evaluate_multiple_choice(self, task_name: str) -> float:
        """Evaluate multiple choice tasks."""
        # Load dataset
        dataset = load_dataset(task_name, split="validation")
        # Handle different dataset types
        if isinstance(dataset, DatasetDict):
            dataset = dataset["validation"]
        dataset = cast(Dataset, dataset)  # Type assertion for ty

        if len(dataset) > 1000:  # Sample for efficiency
            dataset = dataset.select(range(1000))

        correct = 0
        total = 0

        for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
            # Format question
            question = example.get("question", example.get("sentence", ""))
            choices = example.get("choices", {}).get("text", [])

            if not choices:
                continue

            # Get model predictions
            scores = []
            for choice in choices:
                prompt = f"Question: {question}\nAnswer: {choice}"
                score = self._get_sequence_likelihood(prompt)
                scores.append(score)

            # Select best choice
            pred_idx = np.argmax(scores)
            true_idx = example.get("answer", example.get("label", -1))

            if pred_idx == true_idx:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _get_sequence_likelihood(self, text: str) -> float:
        """Calculate likelihood of a sequence."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Calculate average log probability
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return -loss.mean().item()

    def _evaluate_mmlu(self) -> float:
        """Evaluate on MMLU benchmark."""
        # Simplified MMLU evaluation
        # In practice, would evaluate all 57 subjects
        subjects = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
        ]
        scores = []

        for subject in subjects:
            load_dataset("cais/mmlu", subject, split="test")
            score = self._evaluate_multiple_choice(f"mmlu_{subject}")
            scores.append(score)

        return float(np.mean(scores))

    def _evaluate_truthfulqa(self) -> float:
        """Evaluate on TruthfulQA benchmark."""
        # Simplified evaluation - uses multiple choice format
        return self._evaluate_multiple_choice("truthfulqa_mc")

    def _evaluate_gsm8k(self) -> float:
        """Evaluate on GSM8K math benchmark."""
        # Simplified evaluation - in practice would use more sophisticated parsing
        dataset = load_dataset("gsm8k", "main", split="test")
        # Handle different dataset types
        if isinstance(dataset, DatasetDict):
            dataset = dataset["test"]
        dataset = cast(Dataset, dataset)  # Type assertion for ty
        dataset = dataset.select(range(200))  # Sample for efficiency

        correct = 0
        total = 0

        for example in tqdm(dataset, desc="Evaluating GSM8K"):
            question = example["question"]
            answer = example["answer"].split("#### ")[-1].strip()

            # Generate response
            prompt = f"Question: {question}\nLet's think step by step.\n"
            generated = self._generate_text(prompt, max_length=256)

            # Check if answer appears in generation
            if answer in generated:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _generate_text(self, prompt: str, max_length: int = 128) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt) :]
