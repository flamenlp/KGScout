# KGScout Experiment Runner
# All paths are read from dir_mapping.yml
#
# Usage: just k-ablation cwq
#        just k-ablation webqsp

# ============================================================================
# K-ABLATION: Train k=30,50,100,150 → Triplet Analysis → vLLM Inference
# ============================================================================

k-ablation dataset:
    #!/usr/bin/env bash
    set -e

    # --- Read paths from dir_mapping.yml ---
    YAML_OUTPUT=$(python3 scripts/read_config.py "{{dataset}}")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read dir_mapping.yml for dataset '{{dataset}}'"
        exit 1
    fi

    TRAIN=$(echo "$YAML_OUTPUT" | sed -n '1p')
    VAL=$(echo "$YAML_OUTPUT" | sed -n '2p')
    TEST=$(echo "$YAML_OUTPUT" | sed -n '3p')
    K_VALUES=$(echo "$YAML_OUTPUT" | sed -n '4p')
    BASE=$(echo "$YAML_OUTPUT" | sed -n '5p')
    DEFAULT_TOPK=$(echo "$YAML_OUTPUT" | sed -n '6p')
    LLM_MODEL=$(echo "$YAML_OUTPUT" | sed -n '7p')

    LOG="logs/k-ablation.log"
    mkdir -p logs

    echo "============================================================" | tee -a "$LOG"
    echo "K-ABLATION: {{dataset}} | k=$K_VALUES"                        | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    for K in $K_VALUES; do

        echo "" | tee -a "$LOG"
        echo "------------------------------------------------------------" | tee -a "$LOG"
        echo "  K=$K (training, triplet selection, and inference all use top-k=$K)" | tee -a "$LOG"
        echo "------------------------------------------------------------" | tee -a "$LOG"

        # ---- STEP 1: Train model ----
        MODEL_DIR="$BASE/k${K}/model"
        CKPT="$MODEL_DIR/main_training_k${K}/best_model_k${K}.pt"
        if [ -f "$CKPT" ]; then
            echo "  [k=$K] STEP 1: Model exists at $CKPT. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] STEP 1: Training..." | tee -a "$LOG"
            python cli.py train \
                --train-data "$TRAIN" \
                --val-data "$VAL" \
                --checkpoint-dir "$MODEL_DIR" \
                --k $K \
                --num-epochs 30 \
                --early-stopping-patience 10 \
                2>&1 | tee -a "$LOG"
        fi

        # ---- STEP 2: Triplet selection ----
        TRIPLET_DIR="$BASE/k${K}/triplet-analysis"
        TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

        if [ -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] STEP 2: selected_triplets.json exists. Skipping." | tee -a "$LOG"
        elif [ ! -f "$CKPT" ]; then
            echo "  [k=$K] STEP 2: ERROR: Model not found at $CKPT. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] STEP 2: Generating triplets (top-k=$K)..." | tee -a "$LOG"
            python -m src.utils.triplet_selector \
                --model-path "$CKPT" \
                --test-data "$TEST" \
                --output-dir "$TRIPLET_DIR" \
                --top-k $K \
                2>&1 | tee -a "$LOG"
        fi

        # ---- STEP 3: vLLM LLM Inference ----
        RESULT_DIR="$BASE/k${K}/model-result"
        METRICS_FILE="$RESULT_DIR/llm_metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "  [k=$K] STEP 3: LLM results exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] STEP 3: ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] STEP 3: Running vLLM inference (top-k=$K)..." | tee -a "$LOG"
            python run_vllm_inference_ablation.py \
                --input "$TRIPLET_FILE" \
                --output "$RESULT_DIR" \
                --llm-model "$LLM_MODEL" \
                --top-k $K \
                2>&1 | tee -a "$LOG"
        fi

    done

    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "K-ABLATION COMPLETE. Results in: $BASE/"                      | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
