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
LLM_MODEL="llama"

# ---------- RUN ----------
echo "============================================================"
echo "ABLATION-2: Reversed Attention Experiments"
echo "============================================================"

# --- Variant 1: Reversed attention + gated attention weights in combiner ---
# echo ""
# echo ">>> [V1] CWQ..."
# python ablation-2/run_reversed_attention.py \
#     --dataset cwq \
#     --train-data "$CWQ_TRAIN" \
#     --val-data "$CWQ_VAL" \
#     --test-data "$CWQ_TEST" \
#     --output-dir "results/ablation-2" \
#     --llm-model "$LLM_MODEL"
#
# echo ""
# echo ">>> [V1] WebQSP..."
# python ablation-2/run_reversed_attention.py \
#     --dataset webqsp \
#     --train-data "$WEBQSP_TRAIN" \
#     --val-data "$WEBQSP_VAL" \
#     --test-data "$WEBQSP_TEST" \
#     --output-dir "results/ablation-2" \
#     --llm-model "$LLM_MODEL"

# --- Variant 2: Reversed attention + attention weights in tower inputs ---
echo ""
echo ">>> [V2] CWQ..."
python ablation-2/run_reversed_attention2.py \
    --dataset cwq \
    --train-data "$CWQ_TRAIN" \
    --val-data "$CWQ_VAL" \
    --test-data "$CWQ_TEST" \
    --output-dir "results/ablation-2-v2" \
    --llm-model "$LLM_MODEL"

echo ""
echo ">>> [V2] WebQSP..."
python ablation-2/run_reversed_attention2.py \
    --dataset webqsp \
    --train-data "$WEBQSP_TRAIN" \
    --val-data "$WEBQSP_VAL" \
    --test-data "$WEBQSP_TEST" \
    --output-dir "results/ablation-2-v2" \
    --llm-model "$LLM_MODEL"

echo ""
echo "============================================================"
echo "ABLATION-2 COMPLETE."
echo "============================================================"
