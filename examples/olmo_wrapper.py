#!/usr/bin/env python3
"""OLMo wrapper using GPT-NeoX architecture with OLMo configuration."""

import torch
import torch.nn as nn
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in OLMo."""

    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_h_to_4h_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU: (silu(Wx) * Vx)W_out
        gate = self.act(self.dense_h_to_4h(hidden_states))
        up = self.dense_h_to_4h_gate(hidden_states)
        intermediate = gate * up
        output = self.dense_4h_to_h(intermediate)
        return output


class OLMoGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """GPT-NeoX model with OLMo-style configuration and SwiGLU activation."""

    def __init__(self, config):
        super().__init__(config)

        # Replace MLP layers with SwiGLU if configured
        if hasattr(config, "use_swiglu") and config.use_swiglu:
            for layer in self.gpt_neox.layers:
                layer.mlp = SwiGLU(config)

    @classmethod
    def from_olmo_config(cls, olmo_size="4M"):
        """Create model from OLMo configuration presets."""

        # OLMo configurations from the paper
        olmo_configs = {
            "4M": {
                "hidden_size": 64,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 256,
            },
            "6M": {
                "hidden_size": 96,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 384,
            },
            "8M": {
                "hidden_size": 128,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 512,
            },
            "10M": {
                "hidden_size": 144,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 576,
            },
            "14M": {
                "hidden_size": 192,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 768,
            },
            "16M": {
                "hidden_size": 208,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 832,
            },
            "20M": {
                "hidden_size": 192,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 768,
            },
        }

        if olmo_size not in olmo_configs:
            raise ValueError(f"Invalid OLMo size: {olmo_size}. Choose from {list(olmo_configs.keys())}")

        olmo_params = olmo_configs[olmo_size]

        # Create GPT-NeoX config with OLMo parameters
        config = GPTNeoXConfig(
            vocab_size=50277,  # GPT-NeoX tokenizer size
            hidden_size=olmo_params["hidden_size"],
            num_hidden_layers=olmo_params["num_hidden_layers"],
            num_attention_heads=olmo_params["num_attention_heads"],
            intermediate_size=olmo_params["intermediate_size"],
            hidden_act="gelu",  # Will be replaced by SwiGLU
            rotary_pct=0.25,  # Standard RoPE
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=True,
            use_parallel_residual=True,  # GPT-NeoX style
            rope_scaling=None,
        )

        # Add custom flag for SwiGLU
        config.use_swiglu = True

        return cls(config)


def create_olmo_model(size="4M"):
    """Create an OLMo-like model using GPT-NeoX architecture."""
    model = OLMoGPTNeoXForCausalLM.from_olmo_config(size)

    # Initialize weights similar to OLMo
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    model.apply(_init_weights)

    return model


if __name__ == "__main__":
    # Test the model
    print("Creating OLMo 4M model using GPT-NeoX architecture...")
    model = create_olmo_model("4M")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count:,} parameters ({param_count / 1e6:.1f}M)")

    # Test forward pass
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✓ Forward pass successful. Logits shape: {outputs.logits.shape}")

    # Test with labels
    inputs["labels"] = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✓ Loss computation successful: {outputs.loss.item():.4f}")

    # Test generation
    input_text = "The financial markets"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_length=30, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\nGeneration test:")
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")
