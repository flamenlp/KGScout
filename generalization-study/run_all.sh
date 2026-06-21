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
# CONFIGURATION - Update these paths to match your environment
# =============================================================================

# MetaQA raw data paths
KB_PATH="data/metaqa/kb.txt"
QA_1HOP="data/metaqa/1-hop/vanilla/qa_test.txt"
QA_2HOP="data/metaqa/2-hop/vanilla/qa_test.txt"
QA_3HOP="data/metaqa/3-hop/vanilla/qa_test.txt"

# Processed output directory
PROCESSED_DIR="data/metaqa/processed"

# KGScout model checkpoint (trained on WebQSP or CWQ)
MODEL_PATH="checkpoints/webqsp-k100/main/"
DATASET_NAME="webqsp"  # "webqsp" or "cwq" (which dataset the model was trained on)

# Evaluation output
OUTPUT_DIR="results/generalization"
COVERAGE_OUTPUT_DIR="results/generalization/coverage"

# Parameters
TOP_K=100
MAX_TRIPLETS=1000
LLM_MODEL="llama"  # "llama", "qwen", or "deepseek"
EMBEDDING_MODEL="all-MiniLM-L6-v2"

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
