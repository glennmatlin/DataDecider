#!/bin/bash
# Safe execution wrapper for FinPile tokenization with automatic recovery

set -e  # Exit on error

# Configuration
SCRIPT_DIR="/home/gmatlin/Codespace/DataDecider"
INPUT_DIR="/mnt/z/FinPile/0fp-100dolma"
OUTPUT_DIR="/mnt/z/FinPile/tokenized/0fp-100dolma"
LOG_DIR="${OUTPUT_DIR}/logs"
STOP_FILE="${OUTPUT_DIR}/STOP"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to check disk space
check_disk_space() {
    local available=$(df /mnt/z | awk 'NR==2 {print $4}')
    local available_gb=$((available / 1024 / 1024))
    
    if [ $available_gb -lt 200 ]; then
        print_error "Insufficient disk space: ${available_gb}GB available (need at least 200GB)"
        return 1
    else
        print_status "Disk space OK: ${available_gb}GB available"
        return 0
    fi
}

# Function to check if process should stop
should_stop() {
    if [ -f "$STOP_FILE" ]; then
        print_warning "STOP file detected. Gracefully stopping..."
        rm -f "$STOP_FILE"
        return 0
    fi
    return 1
}

# Function to get current progress
get_progress() {
    if [ -f "${OUTPUT_DIR}/processing_state.json" ]; then
        local processed=$(jq -r '.processed_files | length' "${OUTPUT_DIR}/processing_state.json" 2>/dev/null || echo "0")
        local total=200
        local pct=$((processed * 100 / total))
        echo "Progress: ${processed}/${total} files (${pct}%)"
    else
        echo "Progress: Not started"
    fi
}

# Main execution
print_status "Starting FinPile tokenization"
print_status "Input: $INPUT_DIR"
print_status "Output: $OUTPUT_DIR"
print_status "Logs: $LOG_DIR"
echo ""

# Pre-flight checks
print_status "Running pre-flight checks..."

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    print_error "Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check disk space
if ! check_disk_space; then
    exit 1
fi

# Count input files
NUM_FILES=$(ls -1 "$INPUT_DIR"/*.json.gz 2>/dev/null | wc -l)
print_status "Found $NUM_FILES input files"

if [ $NUM_FILES -eq 0 ]; then
    print_error "No .json.gz files found in input directory"
    exit 1
fi

echo ""
print_status "Pre-flight checks passed. Starting tokenization..."
echo ""

# Create log file with timestamp
LOG_FILE="${LOG_DIR}/tokenization_$(date +%Y%m%d_%H%M%S).log"
print_status "Logging to: $LOG_FILE"

# Main tokenization loop with automatic restart
RESTART_COUNT=0
MAX_RESTARTS=10

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    # Check if we should stop
    if should_stop; then
        print_warning "Stopping as requested"
        break
    fi
    
    # Check disk space before each attempt
    if ! check_disk_space; then
        print_error "Stopping due to insufficient disk space"
        break
    fi
    
    # Print current progress
    print_status "$(get_progress)"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run tokenization with nice priority
    print_status "Starting tokenization (attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)..."
    
    nice -n 10 uv run python scripts/tokenize_finpile_enhanced.py \
        --input-path "$INPUT_DIR" \
        --output-path "$OUTPUT_DIR" \
        --tokenizer "EleutherAI/gpt-neox-20b" \
        --max-seq-length 2048 \
        --batch-size 1000 \
        --validation-split 0.05 \
        --num-proc 16 \
        --save-format arrow \
        --checkpoint-interval 10 \
        --memory-limit 50 \
        --verify-samples 1000 \
        --error-tolerance 0.01 \
        2>&1 | tee -a "$LOG_FILE"
    
    # Check exit code
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_status "Tokenization completed successfully!"
        break
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        print_warning "Tokenization failed with exit code $EXIT_CODE"
        
        if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
            print_status "Will restart in 30 seconds..."
            sleep 30
        else
            print_error "Maximum restart attempts reached. Exiting."
            exit 1
        fi
    fi
done

# Final summary
echo ""
print_status "Final status:"
print_status "$(get_progress)"

# Check if final dataset exists
if [ -d "${OUTPUT_DIR}/final" ]; then
    print_status "Final dataset found at: ${OUTPUT_DIR}/final"
    
    # Get dataset size
    DATASET_SIZE=$(du -sh "${OUTPUT_DIR}/final" | cut -f1)
    print_status "Dataset size: $DATASET_SIZE"
else
    print_warning "Final dataset not found. Check logs for errors."
fi

echo ""
print_status "Tokenization script completed"
print_status "Logs saved to: $LOG_FILE"

# Instructions for monitoring
echo ""
echo "To monitor progress in another terminal, run:"
echo "  python scripts/monitor_finpile_tokenization.py --output-path $OUTPUT_DIR"
echo ""
echo "To gracefully stop tokenization, create a STOP file:"
echo "  touch ${OUTPUT_DIR}/STOP"