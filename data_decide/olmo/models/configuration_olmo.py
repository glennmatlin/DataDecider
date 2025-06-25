# src/models/configuration_olmo.py
from transformers import PretrainedConfig
from typing import Optional


class OLMoConfig(PretrainedConfig):
    """Configuration class for OLMo models."""

    model_type = "olmo"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swiglu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        use_bias: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_bias = use_bias
        self.tie_word_embeddings = tie_word_embeddings


# Predefined configurations for each model size
OLMO_CONFIGS = {
    "4M": OLMoConfig(
        vocab_size=50254,  # GPT-NeoX-20B tokenizer
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=256,
    ),
    "6M": OLMoConfig(
        hidden_size=96,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=384,
    ),
    "8M": OLMoConfig(
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=512,
    ),
    "10M": OLMoConfig(
        hidden_size=144,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=576,
    ),
    "14M": OLMoConfig(
        hidden_size=192,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=768,
    ),
    "16M": OLMoConfig(
        hidden_size=208,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=832,
    ),
    "20M": OLMoConfig(
        hidden_size=192,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=768,
    ),
    "60M": OLMoConfig(
        hidden_size=384,
        num_hidden_layers=16,
        num_attention_heads=12,
        intermediate_size=1536,
    ),
    "90M": OLMoConfig(
        hidden_size=528,
        num_hidden_layers=16,
        num_attention_heads=12,
        intermediate_size=2112,
    ),
    "150M": OLMoConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    ),
    "300M": OLMoConfig(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
    ),
    "530M": OLMoConfig(
        hidden_size=1344,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=5376,
    ),
    "750M": OLMoConfig(
        hidden_size=1536,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=6144,
    ),
    "1B": OLMoConfig(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=8192,
    ),
}
