# Type Safety Guide for DataDecider

This guide documents the type safety improvements made to the codebase and provides guidance for maintaining type safety going forward.

## Type Checking Tools

We use two type checkers:
- **ty** - A fast, experimental type checker with excellent performance
- **mypy** - The standard Python type checker with comprehensive features

## Current Status

### ✅ Fixed Issues:
1. **unified_tokenizer.py**:
   - Fixed nullable type annotations (`Optional[T]`)
   - Added proper generic type parameters
   - Fixed console.print safety check
   - Added return type annotations

2. **train.py**:
   - Removed invalid `callbacks` parameter from OLMoTrainer

3. **evaluator.py**:
   - Fixed dataset type handling
   - Added proper type casting for numpy returns
   - Improved Dataset/DatasetDict handling

## Common Type Patterns

### 1. Optional Parameters
```python
# ❌ Bad
def process(data: dict = None):
    pass

# ✅ Good
from typing import Optional, Dict, Any

def process(data: Optional[Dict[str, Any]] = None):
    pass
```

### 2. Generic Types
```python
# ❌ Bad
def get_config() -> Dict:
    return {}

# ✅ Good
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    return {}
```

### 3. Union Types
```python
# ❌ Bad
compression: str = None  # Can be string or None

# ✅ Good
from typing import Optional, Literal

compression: Optional[Literal["gzip", "snappy", "zstd"]] = None
```

### 4. Type Assertions
When you know more than the type checker:
```python
from typing import cast
from datasets import Dataset, DatasetDict

dataset = load_dataset("task", split="validation")
if isinstance(dataset, DatasetDict):
    dataset = dataset["validation"]
dataset = cast(Dataset, dataset)  # Assert type for type checker
```

## Running Type Checks

### Using ty (fast, recommended for development):
```bash
# Check single file
uv run ty check data_decide/scripts/unified_tokenizer.py

# Check entire module
uv run ty check data_decide/
```

### Using mypy (comprehensive):
```bash
# Check entire project
uv run mypy data_decide/

# Check single file
uv run mypy data_decide/scripts/unified_tokenizer.py
```

## CI/CD Integration

Add to your GitHub Actions workflow:
```yaml
- name: Type Check with ty
  run: |
    uv pip install -e ".[dev]"
    uv run ty check data_decide/

- name: Type Check with mypy
  run: |
    uv run mypy data_decide/
```

## Best Practices

1. **Always add type hints to new functions**:
   ```python
   def process_data(
       input_path: Path,
       output_path: Path,
       batch_size: int = 100
   ) -> Dict[str, Any]:
       pass
   ```

2. **Use descriptive type aliases**:
   ```python
   from typing import TypeAlias

   TokenSequence: TypeAlias = List[int]
   TokenBatch: TypeAlias = List[TokenSequence]
   ```

3. **Avoid `Any` when possible**:
   ```python
   # ❌ Avoid
   def process(data: Any) -> Any:
       pass

   # ✅ Better
   def process(data: Union[str, Dict[str, str]]) -> Optional[str]:
       pass
   ```

4. **Document complex types**:
   ```python
   from typing import TypedDict

   class TrainingConfig(TypedDict):
       """Configuration for training."""
       learning_rate: float
       batch_size: int
       num_epochs: int
       warmup_steps: Optional[int]
   ```

## Type Stubs

For libraries without type hints, install type stubs:
```bash
uv pip install types-requests types-pyyaml
```

## Future Improvements

1. **Gradual typing**: Start with critical paths and expand coverage
2. **Type protocols**: Define interfaces for better abstraction
3. **Runtime validation**: Consider pydantic for config validation
4. **Strict mode**: Eventually enable stricter mypy settings

## Common Issues and Solutions

### Issue: "Missing type parameters"
```python
# ty error: Generic type requires parameters
stats: Dict = {}

# Fix: Add type parameters
stats: Dict[str, Union[int, float]] = {}
```

### Issue: "Optional not imported"
```python
# Error: name 'Optional' is not defined
def process(data: Optional[str] = None):

# Fix: Import from typing
from typing import Optional
```

### Issue: "Incompatible return type"
```python
# Error: Expected float, got np.float64
return np.mean(values)

# Fix: Cast to expected type
return float(np.mean(values))
```

## Resources

- [ty documentation](https://github.com/astral-sh/ty)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Python typing documentation](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
