# KGScout Experiment Runner
# Usage: just k-ablation cwq
#        just k-ablation webqsp

# ============================================================================
# DATA PATHS
# ============================================================================

cwq_train := "/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/train/train_jointrainer_path_dataset_v3_ppr.pt"
cwq_val   := "/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/val/val_jointrainer_path_dataset_v3_ppr.pt"
cwq_test  := "/mnt/LS226/LS25/sourav23099/cwq/cwq-rml-v2/test/test_jointrainer_path_dataset_v3_ppr.pt"

webqsp_train := "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/train/train_jointrainer_path_dataset_v3_ppr.pt"
webqsp_val   := "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/val/val_jointrainer_path_dataset_v3_ppr.pt"
webqsp_test  := "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/test/test_jointrainer_path_dataset_v3_ppr.pt"

# ============================================================================
# K-ABLATION: Train k=30,50,100,150 → Triplet Analysis → LLM Inference
# ============================================================================

k-ablation dataset:
    #!/usr/bin/env bash
    set -e

    if [ "{{dataset}}" = "cwq" ]; then
        TRAIN="{{cwq_train}}"
        VAL="{{cwq_val}}"
        TEST="{{cwq_test}}"
    elif [ "{{dataset}}" = "webqsp" ]; then
        TRAIN="{{webqsp_train}}"
        VAL="{{webqsp_val}}"
        TEST="{{webqsp_test}}"
    else
        echo "ERROR: Invalid dataset '{{dataset}}'. Use 'cwq' or 'webqsp'."
        exit 1
    fi

    BASE="results/k-ablation"
    LOG="logs/k-ablation.log"
    mkdir -p logs

    echo "============================================================" | tee -a "$LOG"
    echo "K-ABLATION: {{dataset}} | k=30,50,100,150"                    | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # ---- STEP 1: Train models for each k ----
    echo "" | tee -a "$LOG"
    echo ">>> STEP 1: Training models..." | tee -a "$LOG"

    for K in 30 50 100 150; do
        MODEL_DIR="$BASE/k${K}/model"
        CKPT="$MODEL_DIR/main_training_k${K}/best_model_k${K}.pt"
        if [ -f "$CKPT" ]; then
            echo "  [k=$K] Model exists at $CKPT. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] Training..." | tee -a "$LOG"
            python cli.py train \
                --train-data "$TRAIN" \
                --val-data "$VAL" \
                --checkpoint-dir "$MODEL_DIR" \
                --k $K \
                --num-epochs 30 \
                --early-stopping-patience 10 \
                2>&1 | tee -a "$LOG"
        fi
    done

    # ---- STEP 2: Triplet analysis for each k ----
    echo "" | tee -a "$LOG"
    echo ">>> STEP 2: Triplet analysis..." | tee -a "$LOG"

    for K in 30 50 100 150; do
        TRIPLET_DIR="$BASE/k${K}/triplet-analysis"
        TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"
        CKPT="$BASE/k${K}/model/main_training_k${K}/best_model_k${K}.pt"

        if [ -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] selected_triplets.json exists. Skipping." | tee -a "$LOG"
        elif [ ! -f "$CKPT" ]; then
            echo "  [k=$K] ERROR: Model not found at $CKPT. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] Generating triplets..." | tee -a "$LOG"
            python -m src.utils.triplet_selector \
                --model-path "$CKPT" \
                --test-data "$TEST" \
                --output-dir "$TRIPLET_DIR" \
                --top-k $K \
                2>&1 | tee -a "$LOG"
        fi
    done

    # ---- STEP 3: LLM Inference for each k ----
    echo "" | tee -a "$LOG"
    echo ">>> STEP 3: LLM Inference..." | tee -a "$LOG"

    for K in 30 50 100 150; do
        RESULT_DIR="$BASE/k${K}/model-result"
        METRICS_FILE="$RESULT_DIR/llm_metrics.json"
        TRIPLET_FILE="$BASE/k${K}/triplet-analysis/selected_triplets.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "  [k=$K] LLM results exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] Running LLM inference..." | tee -a "$LOG"
            python ablation-2/run_inference.py \
                --input "$TRIPLET_FILE" \
                --output "$RESULT_DIR" \
                --llm-model llama \
                --top-k $K \
                2>&1 | tee -a "$LOG"
        fi
    done

    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "K-ABLATION COMPLETE. Results in: $BASE/"                      | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
