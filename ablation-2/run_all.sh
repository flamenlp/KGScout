#!/bin/bash
# =============================================================================
# Ablation-2: Reversed Attention Architecture Study
# Runs for both CWQ and WebQSP datasets sequentially.
# =============================================================================

set -e

# ---------- DATA PATHS ----------
# CWQ
CWQ_TRAIN="/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/train/train_jointrainer_path_dataset_v3_ppr.pt"
CWQ_VAL="/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/val/val_jointrainer_path_dataset_v3_ppr.pt"
CWQ_TEST="/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/test/test_jointrainer_path_dataset_v3_ppr.pt"

# WebQSP
WEBQSP_TRAIN="/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/train/train_jointrainer_path_dataset_v3_ppr.pt"
WEBQSP_VAL="/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/val/val_jointrainer_path_dataset_v3_ppr.pt"
WEBQSP_TEST="/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/test/test_jointrainer_path_dataset_v3_ppr.pt"

# ---------- OUTPUT ----------
OUTPUT_DIR="./results/ablation-2"
LLM_MODEL="llama"

# ---------- RUN ----------
echo "============================================================"
echo "ABLATION-2: Reversed Attention (Q=question, K=V=triplets)"
echo "============================================================"

echo ""
echo ">>> Running CWQ..."
python ablation-2/run_reversed_attention.py \
    --dataset cwq \
    --train-data "$CWQ_TRAIN" \
    --val-data "$CWQ_VAL" \
    --test-data "$CWQ_TEST" \
    --output-dir "$OUTPUT_DIR" \
    --llm-model "$LLM_MODEL"

echo ""
echo ">>> Running WebQSP..."
python ablation-2/run_reversed_attention.py \
    --dataset webqsp \
    --train-data "$WEBQSP_TRAIN" \
    --val-data "$WEBQSP_VAL" \
    --test-data "$WEBQSP_TEST" \
    --output-dir "$OUTPUT_DIR" \
    --llm-model "$LLM_MODEL"

echo ""
echo "============================================================"
echo "ABLATION-2 COMPLETE. Results in: $OUTPUT_DIR"
echo "============================================================"
