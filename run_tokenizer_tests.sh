#!/bin/bash
# Script to run tokenization tests with coverage

echo "Running tokenization system tests..."
echo "===================================="

# Run all tokenization tests with coverage
uv run pytest tests/test_*tokenizer*.py tests/test_tokenizer_*.py \
    -v \
    --cov=data_decide.scripts.unified_tokenizer \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-branch \
    --no-cov-on-fail \
    -x

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
    echo "Open with: xdg-open htmlcov/index.html"
else
    echo ""
    echo "❌ Some tests failed. Please check the output above."
fi
