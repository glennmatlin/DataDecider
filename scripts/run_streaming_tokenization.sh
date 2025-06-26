#!/bin/bash
# Run streaming tokenization with monitoring

set -e

OUTPUT_DIR="/mnt/z/FinPile/tokenized/0fp-100dolma_streaming"
LOG_FILE="$OUTPUT_DIR/tokenization_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"

echo "Starting streaming tokenization..."
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"

nohup uv run python scripts/tokenize_finpile_streaming.py \
    --input-dir /mnt/z/FinPile/0fp-100dolma \
    --output-dir "$OUTPUT_DIR" \
    --tokenizer EleutherAI/gpt-neox-20b \
    --max-seq-length 2048 \
    --sequences-per-chunk 10000 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started with PID: $PID"
echo "Monitor with: tail -f $LOG_FILE"