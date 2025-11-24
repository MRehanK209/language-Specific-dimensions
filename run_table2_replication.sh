#!/bin/bash
###############################################################################
# Full Table 2 Replication Script
# 
# This script replicates Table 2 results for:
# - Models: Llama-2-7B, Llama-2-13B, Llama-3.1-8B
# - Languages: French, German, Spanish, Chinese, Japanese  
# - Settings: Monolingual and Parallel
# - 3 runs with different random seeds per experiment
#
# Usage:
#   nohup bash run_table2_replication.sh > table2_replication.log 2>&1 &
###############################################################################

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Set Hugging Face token (if needed)
# export HF_TOKEN="your_token_here"

# Output directory
OUTPUT_DIR="results/table2_replication"
mkdir -p "$OUTPUT_DIR"

# Models to test (from config.py)
MODELS=(
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Meta-Llama-3.1-8B"
)

# Languages to test (FLORES-200 codes)
LANGUAGES=(
    "fra_Latn"   # French
    "deu_Latn"   # German
    "spa_Latn"   # Spanish
    "zho_Hans"   # Chinese
    "jpn_Jpan"   # Japanese
)

# Settings
SETTINGS=(
    "monolingual"
    "parallel"
)

# Number of runs (paper uses 3)
NUM_RUNS=3

###############################################################################
# Main execution
###############################################################################

echo "================================================================================"
echo "TABLE 2 REPLICATION - FULL EXPERIMENT"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Models: ${MODELS[@]}"
echo "  Languages: ${LANGUAGES[@]}"
echo "  Settings: ${SETTINGS[@]}"
echo "  Runs per experiment: $NUM_RUNS"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Total experiments: $((${#MODELS[@]} * ${#LANGUAGES[@]} * ${#SETTINGS[@]}))"
echo "================================================================================"
echo ""

# Run all experiments in one go
python -u replicate_table2.py \
    --models "${MODELS[@]}" \
    --languages "${LANGUAGES[@]}" \
    --settings "${SETTINGS[@]}" \
    --num_runs "$NUM_RUNS" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "================================================================================"
echo "TABLE 2 REPLICATION COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================================================"

