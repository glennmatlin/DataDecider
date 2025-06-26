# FinPileCode Migration Guide

This document provides context and guidance for continuing OLMo model development that was started in the FinPileCode repository.

## Quick Start

1. **Check Applied Fixes**:
   ```bash
   # Verify inv_freq fix is applied
   grep -n "inv_freq" data_decide/olmo/models/olmo_model.py
   # Should show: inv_freq = 1.0 / ... (not self.inv_freq)

   # Verify vocab size fix
   grep -n "vocab_size=50277" data_decide/olmo/models/configuration_olmo.py
   ```

2. **Test Model Creation**:
   ```bash
   cd scripts/debug
   python test_olmo_invfreq.py
   # Should output: "✓ Model created successfully!"
   ```

3. **Run Working Training Example**:
   ```bash
   cd examples
   python train_olmo_gpu.py
   # Uses GPT-NeoX fallback, achieves ~1800 perplexity
   ```

## Migration Summary

### What Was Done
- Fixed `inv_freq` double assignment bug in RotaryEmbedding
- Fixed vocabulary size mismatch (50254 → 50277)
- Created multiple training scripts with GPU optimization
- Implemented PerplexityEvaluator for model evaluation
- Developed OLMo wrapper using GPT-NeoX architecture

### What Needs Fixing
1. **Position Embeddings** - apply_rotary_pos_emb uses indexing instead of broadcasting
2. **Attention Mask** - needs 4D reshaping for multi-head attention

### Files Migrated

#### Debug Scripts (`scripts/debug/`)
- `test_olmo_invfreq.py` - Tests the inv_freq fix
- `debug_olmo_cuda.py` - CUDA error debugging with detailed tracking
- `test_olmo_training.py` - Basic training loop test

#### Training Examples (`examples/`)
- `train_olmo_gpu.py` - Working GPU training with perplexity evaluation
- `olmo_wrapper.py` - OLMo-like model using GPT-NeoX base
- `train_olmo_wrapper_gpu.py` - Training script for the wrapper

## Critical Fixes to Apply

### 1. Fix Position Embeddings

In `data_decide/olmo/models/olmo_model.py`, update the `apply_rotary_pos_emb` function:

```python
# Current (BROKEN)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(0)  # [seq_len, dim]
    q_embed = (q * cos[position_ids]) + (rotate_half(q) * sin[position_ids])
    k_embed = (k * cos[position_ids]) + (rotate_half(k) * sin[position_ids])
    return q_embed, k_embed

# Fixed (TODO)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings using broadcasting."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 2. Fix Attention Mask

In `OLMoAttention.forward()` method, before line 107:

```python
# Add this before: attn_weights = attn_weights + attention_mask
if attention_mask is not None:
    # Ensure mask is 4D: [batch, num_heads, seq_len, seq_len]
    if attention_mask.dim() == 2:
        # From [batch, seq_len] to [batch, 1, 1, seq_len]
        attention_mask = attention_mask[:, None, None, :]
    if attention_mask.dim() == 3:
        # From [batch, 1, seq_len] to [batch, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1)
    # Expand to match attention weights shape
    attention_mask = attention_mask.expand(bsz, self.num_heads, q_len, -1)
```

## Working Configurations

### Model Configuration
```python
from data_decide.olmo.models import OLMoConfig

config = OLMoConfig(
    vocab_size=50277,  # MUST be 50277 for GPT-NeoX tokenizer
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=256,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    layer_norm_eps=1e-5
)
```

### Training Hyperparameters
```python
BATCH_SIZE = 8  # Works on 8GB GPU
LEARNING_RATE = 1.4e-2  # From OLMo paper
MAX_STEPS = 100  # For testing
FP16 = True  # Essential for memory
```

## Performance Benchmarks

| Model | Parameters | Val Perplexity | GPU Memory | Status |
|-------|------------|----------------|------------|---------|
| GPT-NeoX (first) | 6.8M | 8,882 | 1.39 GB | ✓ Working |
| GPT-NeoX (optimized) | 6.8M | 1,817 | 2.68 GB | ✓ Working |
| OLMo 4M (actual) | 3.7M | TBD | TBD | ❌ Needs fixes |
| OLMo Wrapper | 3.7M | TBD | TBD | 🔧 Ready to test |

## Next Steps Priority

1. **Immediate** - Test the wrapper approach:
   ```bash
   cd examples
   python train_olmo_wrapper_gpu.py
   ```

2. **High Priority** - Fix RoPE implementation:
   - Apply broadcasting fix to apply_rotary_pos_emb
   - Test with debug_olmo_cuda.py

3. **High Priority** - Fix attention mask:
   - Add reshaping code to OLMoAttention
   - Test with full batch training

4. **Medium Priority** - Download full dataset:
   - Need 20 arXiv shards (listed in configs/data_configs/arxiv_shards.txt)
   - ~9.6GB total

5. **Low Priority** - Production training:
   - Use distributed training for full 400M tokens
   - Target matching OLMo paper results

## Debugging Tips

1. **For CUDA errors**:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python your_script.py
   ```

2. **Check GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Test in isolation**:
   ```python
   # Always test with single sample first
   model.eval()
   with torch.no_grad():
       output = model(input_ids)
   ```

4. **Use the debug scripts**:
   - `test_olmo_invfreq.py` - Quick model creation test
   - `debug_olmo_cuda.py` - Detailed CUDA debugging
   - `test_olmo_training.py` - Simple training loop

## Common Issues and Solutions

| Issue | Error Message | Solution |
|-------|--------------|----------|
| inv_freq | "attribute 'inv_freq' already exists" | Use local variable, not self.inv_freq |
| Vocab size | "size mismatch for embed_tokens.weight" | Set vocab_size=50277 |
| CUDA index | "srcIndex < srcSelectDimSize" | Fix position_ids indexing |
| Tensor shape | "size of tensor a (512) must match..." | Fix attention mask dimensions |

## Contact and Resources

- Original work done in: `/home/gmatlin/Codespace/FinPileCode`
- Full context available in: `DATADECIDER_CONTEXT.md`
- OLMo paper: [arXiv link would go here]
- DataDecide methodology: See citations/DataDecide.htm

## Session Information

- **Date**: 2025-06-25
- **Duration**: ~8 hours
- **Models trained**: 2 (GPT-NeoX variants)
- **Best result**: 1,817 perplexity on validation
- **GPU used**: NVIDIA GeForce RTX 3070 Ti (8GB)
