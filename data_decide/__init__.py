"""
data-decide: OLMo training and evaluation framework using DataDecide methodology.

This package provides tools for training and evaluating Open Language Models (OLMo)
using the DataDecide approach for efficient data curation and model development.
"""

__version__ = "0.1.0"

# Re-export key components
from . import olmo

__all__ = ["olmo", "__version__"]