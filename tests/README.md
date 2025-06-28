# Tokenization System Tests

This directory contains comprehensive tests for the unified tokenization system.

## Test Structure

- `conftest.py` - Shared pytest fixtures and utilities
- `test_unified_tokenizer.py` - Integration tests for all tokenization modes
- `test_tokenizer_components.py` - Unit tests for individual components
- `test_tokenizer_data_integrity.py` - Data validation and integrity tests
- `test_data/` - Sample data files for testing

## Running Tests

### Run all tests with coverage:
```bash
uv run pytest
```

### Run specific test categories:
```bash
# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# Exclude slow tests
uv run pytest -m "not slow"
```

### Run specific test files:
```bash
uv run pytest tests/test_unified_tokenizer.py
uv run pytest tests/test_tokenizer_components.py::TestTokenizationConfig
```

### Generate coverage report:
```bash
uv run pytest --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Coverage Goals

- **Statements**: 85%+
- **Branches**: 80%+
- **Functions**: 90%+

## Key Test Scenarios

### Integration Tests
- All three processing modes (batch, streaming, hybrid)
- Parallel processing with multiple workers
- Checkpoint/resume functionality
- Different file formats (JSON, JSONL, GZ, text)
- Error handling and recovery

### Component Tests
- Configuration validation
- Statistics tracking
- Checkpoint management
- Monitoring functionality
- Standalone worker functions

### Data Integrity Tests
- Sequence length handling
- EOS token appending
- Validation splits
- Checksum verification
- Metadata generation

## Performance Benchmarks

The tests include performance assertions to ensure:
- Hybrid mode is faster than streaming mode
- Memory limits are enforced in streaming mode
- Parallel processing improves throughput

## Adding New Tests

1. Use appropriate fixtures from `conftest.py`
2. Mark tests with appropriate markers (`@pytest.mark.unit`, etc.)
3. Follow the AAA pattern (Arrange, Act, Assert)
4. Include docstrings explaining what is being tested
5. Test both success and failure cases
