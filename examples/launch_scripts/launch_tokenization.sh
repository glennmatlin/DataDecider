#!/bin/bash
# Production tokenization launcher for FinPile dataset
# This script sets up and launches the unified tokenizer with optimal settings

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
INPUT_DIR="/mnt/z/FinPile/datamix/lda/0fp-100dolma"
OUTPUT_DIR="data/tokenized/finpile_lda"
CHECKPOINT_DIR="tokenization_checkpoints"
LOG_FILE="tokenization_$(date +%Y%m%d_%H%M%S).log"

# Default settings (can be overridden by command line args)
MODE="hybrid"
NUM_WORKERS=4
MAX_MEMORY_GB=16
TOKENIZER="EleutherAI/gpt-neox-20b"
MAX_SEQ_LENGTH=2048
BATCH_SIZE=200

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        print_error "Input directory not found: $INPUT_DIR"
        exit 1
    fi

    # Check if we have the tokenizer script
    if [ ! -f "scripts/tokenize_unified.py" ]; then
        print_error "Unified tokenizer script not found!"
        exit 1
    fi

    # Check Python dependencies
    if ! uv run python -c "import transformers, pyarrow, psutil" 2>/dev/null; then
        print_error "Required Python packages not installed!"
        print_info "Run: uv pip install transformers pyarrow psutil"
        exit 1
    fi

    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "logs"

    print_success "Prerequisites check passed"
}

# Function to estimate time and resources
estimate_requirements() {
    print_info "Estimating resource requirements..."

    # Count input files
    NUM_FILES=$(find "$INPUT_DIR" -name "*.json.gz" | wc -l)

    # Estimate based on previous benchmarks
    # ~4,300 docs/sec in hybrid mode, ~285K docs per file
    DOCS_PER_FILE=285000
    TOTAL_DOCS=$((NUM_FILES * DOCS_PER_FILE))
    DOCS_PER_SEC=4300
    EST_TIME_SEC=$((TOTAL_DOCS / DOCS_PER_SEC / NUM_WORKERS))
    EST_TIME_HOURS=$(echo "scale=1; $EST_TIME_SEC / 3600" | bc)

    # Estimate output size (38.7B tokens, ~77GB)
    EST_OUTPUT_GB=$(echo "scale=1; $NUM_FILES * 0.385" | bc)

    echo
    echo "=== Tokenization Estimates ==="
    echo "Input files: $NUM_FILES"
    echo "Estimated documents: $(echo $TOTAL_DOCS | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
    echo "Workers: $NUM_WORKERS"
    echo "Estimated time: ${EST_TIME_HOURS} hours"
    echo "Estimated output size: ${EST_OUTPUT_GB} GB"
    echo "============================="
    echo
}

# Function to show current progress if resuming
check_existing_progress() {
    if [ -f "$CHECKPOINT_DIR/progress.json" ]; then
        print_info "Found existing checkpoint, checking progress..."

        # Extract some basic info from checkpoint
        COMPLETED_FILES=$(uv run python -c "
import json
with open('$CHECKPOINT_DIR/progress.json', 'r') as f:
    data = json.load(f)
    print(len(data.get('completed_files', [])))
" 2>/dev/null || echo "0")

        if [ "$COMPLETED_FILES" -gt 0 ]; then
            print_warning "Resuming from checkpoint: $COMPLETED_FILES files already processed"
            read -p "Continue from checkpoint? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                read -p "Delete checkpoint and start fresh? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "$CHECKPOINT_DIR"
                    mkdir -p "$CHECKPOINT_DIR"
                    print_info "Checkpoint deleted, starting fresh"
                else
                    print_info "Exiting without changes"
                    exit 0
                fi
            fi
        fi
    fi
}

# Function to launch tokenization
launch_tokenization() {
    print_info "Launching tokenization process..."

    # Build command
    CMD="uv run python scripts/tokenize_unified.py"
    CMD="$CMD $INPUT_DIR"
    CMD="$CMD $OUTPUT_DIR"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --num-workers $NUM_WORKERS"
    CMD="$CMD --max-memory-gb $MAX_MEMORY_GB"
    CMD="$CMD --tokenizer $TOKENIZER"
    CMD="$CMD --max-seq-length $MAX_SEQ_LENGTH"
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
    CMD="$CMD --output-format parquet"
    CMD="$CMD --compression snappy"

    # Add any additional arguments passed to script
    if [ $# -gt 0 ]; then
        CMD="$CMD $@"
    fi

    print_info "Command: $CMD"
    echo

    # Launch with logging
    print_info "Starting tokenization (log: logs/$LOG_FILE)"
    print_info "Press Ctrl+C to stop gracefully (progress will be saved)"
    echo

    # Run tokenization
    $CMD 2>&1 | tee "logs/$LOG_FILE"

    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "Tokenization completed successfully!"
    else
        print_error "Tokenization failed or was interrupted"
        print_info "Check logs/$LOG_FILE for details"
        print_info "Progress was saved and can be resumed"
    fi
}

# Function to show monitoring instructions
show_monitoring_instructions() {
    echo
    echo "=== Monitoring Instructions ==="
    echo "To monitor progress in another terminal, run:"
    echo
    echo "  python scripts/monitor_progress.py $OUTPUT_DIR --checkpoint-dir $CHECKPOINT_DIR --continuous"
    echo
    echo "Or for simple one-time check:"
    echo
    echo "  python scripts/monitor_progress.py $OUTPUT_DIR --checkpoint-dir $CHECKPOINT_DIR"
    echo "==============================="
    echo
}

# Main execution
main() {
    print_info "FinPile Tokenization Launcher"
    echo

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --workers)
                NUM_WORKERS="$2"
                shift 2
                ;;
            --memory)
                MAX_MEMORY_GB="$2"
                shift 2
                ;;
            --tokenizer)
                TOKENIZER="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --mode MODE          Processing mode (batch/streaming/hybrid, default: hybrid)"
                echo "  --workers NUM        Number of workers (default: 4)"
                echo "  --memory GB          Max memory in GB (default: 16)"
                echo "  --tokenizer NAME     Tokenizer name (default: allenai/OLMo-1B)"
                echo "  --help               Show this help"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done

    # Run steps
    check_prerequisites
    check_existing_progress
    estimate_requirements
    show_monitoring_instructions

    # Confirm before starting
    read -p "Ready to start tokenization? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        launch_tokenization "$@"
    else
        print_info "Tokenization cancelled"
    fi
}

# Run main function
main "$@"
