#!/bin/bash
# =============================================================================
# MetaQA Generalization Study - 2-hop and 3-hop ONLY
# =============================================================================

# Script continues on error so the terminal stays open

# =============================================================================
# CONFIGURATION
# =============================================================================

KB_PATH="data/metaqa/kb.txt"
QA_2HOP="data/metaqa/2-hop/vanilla/qa_test.txt"
QA_3HOP="data/metaqa/3-hop/vanilla/qa_test.txt"

PROCESSED_DIR="data/metaqa/processed"

MODEL_PATH="checkpoints/webqsp-k100/main/"
DATASET_NAME="webqsp"

OUTPUT_DIR="results/generalization"
COVERAGE_OUTPUT_DIR="results/generalization/coverage"

TOP_K=100
MAX_TRIPLETS=1000
LLM_MODEL="llama"
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# =============================================================================
# STEP 1: Preprocess 2-hop and 3-hop
# =============================================================================

echo "============================================================"
echo "STEP 1: Preprocessing MetaQA 2-hop and 3-hop"
echo "============================================================"

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
# STEP 2: Coverage-only evaluation (2-hop and 3-hop)
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 2: Coverage evaluation (KGScout vs Cosine baseline)"
echo "============================================================"

python generalization-study/run_coverage_only.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$PROCESSED_DIR" \
    --hop 2 \
    --top-k "$TOP_K" \
    --output-dir "$COVERAGE_OUTPUT_DIR"

python generalization-study/run_coverage_only.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$PROCESSED_DIR" \
    --hop 3 \
    --top-k "$TOP_K" \
    --output-dir "$COVERAGE_OUTPUT_DIR"

# =============================================================================
# STEP 3: Full evaluation with LLM (2-hop and 3-hop)
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 3: Full evaluation (KGScout + LLM + metrics)"
echo "============================================================"

python generalization-study/run_generalization.py \
    --model-path "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --hop 2 \
    --data-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --top-k "$TOP_K" \
    --llm-model "$LLM_MODEL"

python generalization-study/run_generalization.py \
    --model-path "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --hop 3 \
    --data-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --top-k "$TOP_K" \
    --llm-model "$LLM_MODEL"

echo ""
echo "============================================================"
echo "DONE! Results saved to: $OUTPUT_DIR"
echo "============================================================"
