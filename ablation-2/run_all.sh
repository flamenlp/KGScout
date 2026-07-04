#!/bin/bash
# =============================================================================
# Ablation-2: Reversed Attention Architecture Study
# Runs for both CWQ and WebQSP datasets sequentially.
#
# Usage:
#   Full training + evaluation:
#     bash ablation-2/run_all.sh
#
#   Evaluation only (provide model checkpoint):
#     bash ablation-2/run_all.sh --model-checkpoint /path/to/checkpoint_dir
# =============================================================================

set -e

# ---------- PARSE OPTIONAL MODEL CHECKPOINT ----------
MODEL_CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-checkpoint)
            MODEL_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---------- DATA PATHS (read from config.yml) ----------
read_yaml_field() {
    python3 -c "
import yaml
with open('config.yml') as f:
    cfg = yaml.safe_load(f)
ds = cfg['datasets']
print(ds['cwq']['train'])
print(ds['cwq']['val'])
print(ds['cwq']['test'])
print(ds['webqsp']['train'])
print(ds['webqsp']['val'])
print(ds['webqsp']['test'])
print(cfg['defaults']['llm_model'])
"
}

YAML_OUT=$(read_yaml_field)
CWQ_TRAIN=$(echo "$YAML_OUT" | sed -n '1p')
CWQ_VAL=$(echo "$YAML_OUT" | sed -n '2p')
CWQ_TEST=$(echo "$YAML_OUT" | sed -n '3p')
WEBQSP_TRAIN=$(echo "$YAML_OUT" | sed -n '4p')
WEBQSP_VAL=$(echo "$YAML_OUT" | sed -n '5p')
WEBQSP_TEST=$(echo "$YAML_OUT" | sed -n '6p')
LLM_MODEL=$(echo "$YAML_OUT" | sed -n '7p')

# ---------- BUILD CHECKPOINT ARG ----------
CHECKPOINT_ARG=""
if [ -n "$MODEL_CHECKPOINT" ]; then
    CHECKPOINT_ARG="--model-checkpoint $MODEL_CHECKPOINT"
    echo "Mode: EVALUATION ONLY (checkpoint: $MODEL_CHECKPOINT)"
else
    echo "Mode: FULL TRAINING + EVALUATION"
fi

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
#     --llm-model "$LLM_MODEL" \
#     $CHECKPOINT_ARG
#
# echo ""
# echo ">>> [V1] WebQSP..."
# python ablation-2/run_reversed_attention.py \
#     --dataset webqsp \
#     --train-data "$WEBQSP_TRAIN" \
#     --val-data "$WEBQSP_VAL" \
#     --test-data "$WEBQSP_TEST" \
#     --output-dir "results/ablation-2" \
#     --llm-model "$LLM_MODEL" \
#     $CHECKPOINT_ARG

# --- Variant 2: Reversed attention + attention weights in tower inputs ---
echo ""
echo ">>> [V2] CWQ..."
python ablation-2/run_reversed_attention2.py \
    --dataset cwq \
    --train-data "$CWQ_TRAIN" \
    --val-data "$CWQ_VAL" \
    --test-data "$CWQ_TEST" \
    --output-dir "results/ablation-2-v2" \
    --llm-model "$LLM_MODEL" \
    $CHECKPOINT_ARG

echo ""
# echo ">>> [V2] WebQSP..."
# python ablation-2/run_reversed_attention2.py \
#     --dataset webqsp \
#     --train-data "$WEBQSP_TRAIN" \
#     --val-data "$WEBQSP_VAL" \
#     --test-data "$WEBQSP_TEST" \
#     --output-dir "results/ablation-2-v2" \
#     --llm-model "$LLM_MODEL" \
#     $CHECKPOINT_ARG

echo ""
echo "============================================================"
echo "ABLATION-2 COMPLETE."
echo "============================================================"
