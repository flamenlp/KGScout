#!/bin/bash
# =============================================================================
# MetaQA Generalization Study - Full Pipeline
# =============================================================================
# This script runs the entire generalization study:
#   1. Preprocess MetaQA 1-hop, 2-hop, 3-hop test sets
#   2. Run coverage-only evaluation (fast, no LLM)
#   3. Run full evaluation with LLM (KGScout re-rank + Llama inference + metrics)
# =============================================================================

# Script continues on error so the terminal stays open

# =============================================================================
# CONFIGURATION - Read from dir_mapping.yml
# =============================================================================

# Read generalization config from dir_mapping.yml
YAML_OUT=$(python3 -c "
import yaml
with open('dir_mapping.yml') as f:
    cfg = yaml.safe_load(f)
g = cfg['generalization']
d = cfg['defaults']
print(g['kb_path'])
print(g['qa_1hop'])
print(g['qa_2hop'])
print(g['qa_3hop'])
print(g['processed_dir'])
print(g['model_path'])
print(g['embedding_model'])
print(cfg['results']['generalization'])
print(d['top_k'])
print(d['sample_k'])
print(d['llm_model'])
")

# MetaQA raw data paths
KB_PATH=$(echo "$YAML_OUT" | sed -n '1p')
QA_1HOP=$(echo "$YAML_OUT" | sed -n '2p')
QA_2HOP=$(echo "$YAML_OUT" | sed -n '3p')
QA_3HOP=$(echo "$YAML_OUT" | sed -n '4p')

# Processed output directory
PROCESSED_DIR=$(echo "$YAML_OUT" | sed -n '5p')

# KGScout model checkpoint (trained on WebQSP or CWQ)
MODEL_PATH=$(echo "$YAML_OUT" | sed -n '6p')
DATASET_NAME="webqsp"  # "webqsp" or "cwq" (which dataset the model was trained on)

# Evaluation output
OUTPUT_DIR=$(echo "$YAML_OUT" | sed -n '8p')
COVERAGE_OUTPUT_DIR="${OUTPUT_DIR}/coverage"

# Parameters
TOP_K=$(echo "$YAML_OUT" | sed -n '9p')
MAX_TRIPLETS=$(echo "$YAML_OUT" | sed -n '10p')
LLM_MODEL=$(echo "$YAML_OUT" | sed -n '11p')
EMBEDDING_MODEL=$(echo "$YAML_OUT" | sed -n '7p')

# =============================================================================
# STEP 1: Preprocess MetaQA test sets
# =============================================================================

echo "============================================================"
echo "STEP 1: Preprocessing MetaQA test sets"
echo "============================================================"

echo ""
echo "--- Preprocessing 1-hop ---"
if [ -f "$PROCESSED_DIR/metaqa-1hop-test.pt" ]; then
    echo "Already exists: $PROCESSED_DIR/metaqa-1hop-test.pt — skipping."
else
    python generalization-study/preprocess_metaqa.py \
        --kb-path "$KB_PATH" \
        --qa-path "$QA_1HOP" \
        --output-dir "$PROCESSED_DIR" \
        --hop 1 \
        --max-triplets "$MAX_TRIPLETS" \
        --embedding-model "$EMBEDDING_MODEL"
fi

echo ""
echo "--- Preprocessing 2-hop ---"
if [ -f "$PROCESSED_DIR/metaqa-2hop-test.pt" ]; then
    echo "Already exists: $PROCESSED_DIR/metaqa-2hop-test.pt — skipping."
else
    python generalization-study/preprocess_metaqa.py \
        --kb-path "$KB_PATH" \
        --qa-path "$QA_2HOP" \
        --output-dir "$PROCESSED_DIR" \
        --hop 2 \
        --max-triplets "$MAX_TRIPLETS" \
        --embedding-model "$EMBEDDING_MODEL"
fi

echo ""
echo "--- Preprocessing 3-hop ---"
if [ -f "$PROCESSED_DIR/metaqa-3hop-test.pt" ]; then
    echo "Already exists: $PROCESSED_DIR/metaqa-3hop-test.pt — skipping."
else
    python generalization-study/preprocess_metaqa.py \
        --kb-path "$KB_PATH" \
        --qa-path "$QA_3HOP" \
        --output-dir "$PROCESSED_DIR" \
        --hop 3 \
        --max-triplets "$MAX_TRIPLETS" \
        --embedding-model "$EMBEDDING_MODEL"
fi

# =============================================================================
# STEP 2: Coverage-only evaluation (fast, no LLM needed)
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 2: Coverage evaluation (KGScout vs Cosine baseline)"
echo "============================================================"

python generalization-study/run_coverage_only.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$PROCESSED_DIR" \
    --all-hops \
    --top-k "$TOP_K" \
    --output-dir "$COVERAGE_OUTPUT_DIR"

# =============================================================================
# STEP 3: Full evaluation with LLM
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 3: Full evaluation (KGScout + LLM + metrics)"
echo "============================================================"

python generalization-study/run_generalization.py \
    --model-path "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --all-hops \
    --data-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --top-k "$TOP_K" \
    --llm-model "$LLM_MODEL"

echo ""
echo "============================================================"
echo "DONE! Results saved to: $OUTPUT_DIR"
echo "============================================================"
