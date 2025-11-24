#!/bin/bash
# Run all paper replication experiments for Llama-2-7B

set -e

echo "========================================"
echo "LLama2-7B Analysis"
echo "========================================"
echo ""

# Run all experiments
python llama2_analysis.py \
    --model meta-llama/Llama-2-7b-hf \
    --experiments all \
    --languages zho_Hans jpn_Jpan fra_Latn spa_Latn deu_Latn \
    --test_language fra_Latn \
    --output_dir results/llama-2-7b-analysis \
    --device cuda

echo ""
echo "========================================"
echo "Complete! Results in:"
echo "  - results/paper_replication/table1_overlap_matrix/"
echo "  - results/paper_replication/figure3_topk_selection/"
echo "  - results/paper_replication/figure4_layer_selection/"
echo "  - results/paper_replication/figure5_overlap_rate/"
echo "========================================"

