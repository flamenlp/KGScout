# KGScout Experiment Runner
# All paths are read from config.yml
#
# Usage: just k-ablation cwq
#        just k-ablation webqsp

# ============================================================================
# K-ABLATION: Train k=30,50,100,150 → Triplet Analysis → vLLM Inference
# ============================================================================

k-ablation dataset:
    #!/usr/bin/env bash
    # --- Read paths from config.yml ---
    YAML_OUTPUT=$(python3 scripts/read_config.py "{{dataset}}")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read config.yml for dataset '{{dataset}}'"
        exit 1
    fi

    TRAIN=$(echo "$YAML_OUTPUT" | sed -n '1p')
    VAL=$(echo "$YAML_OUTPUT" | sed -n '2p')
    TEST=$(echo "$YAML_OUTPUT" | sed -n '3p')
    K_VALUES=$(echo "$YAML_OUTPUT" | sed -n '4p')
    BASE=$(echo "$YAML_OUTPUT" | sed -n '5p')
    DEFAULT_TOPK=$(echo "$YAML_OUTPUT" | sed -n '6p')
    LLM_MODEL=$(echo "$YAML_OUTPUT" | sed -n '7p')

    # Include dataset name in output path
    BASE="$BASE/{{dataset}}"

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


# ============================================================================
# K-ABLATION-COSINE: Cosine baseline (no training) → Triplet Selection → vLLM Inference
# ============================================================================

k-ablation-cosine dataset:
    #!/usr/bin/env bash
    # --- Read paths from config.yml ---
    YAML_OUTPUT=$(python3 scripts/read_config.py "{{dataset}}")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read config.yml for dataset '{{dataset}}'"
        exit 1
    fi

    TEST=$(echo "$YAML_OUTPUT" | sed -n '3p')
    K_VALUES=$(echo "$YAML_OUTPUT" | sed -n '4p')
    LLM_MODEL=$(echo "$YAML_OUTPUT" | sed -n '7p')

    # Cosine ablation base dir with dataset subdirectory
    BASE="./results/cosine-k-ablation/{{dataset}}"

    LOG="logs/cosine-k-ablation.log"
    mkdir -p logs

    echo "============================================================" | tee -a "$LOG"
    echo "K-ABLATION-COSINE: {{dataset}} | k=$K_VALUES"                 | tee -a "$LOG"
    echo "  Retriever: cosine (no trained model)"                       | tee -a "$LOG"
    echo "  Test data: $TEST"                                           | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # ---- PHASE 1: Generate all triplet files ----
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 1: Triplet selection (all k-values)" | tee -a "$LOG"

    for K in $K_VALUES; do
        TRIPLET_DIR="$BASE/k${K}/triplet-analysis"
        TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

        if [ -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] selected_triplets.json exists. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] Generating cosine triplets (top-k=$K)..." | tee -a "$LOG"
            python -m src.utils.triplet_selector \
                --test-data "$TEST" \
                --output-dir "$TRIPLET_DIR" \
                --top-k $K \
                --retriever cosine \
                2>&1 | tee -a "$LOG"
        fi
    done

    # ---- PHASE 2: vLLM LLM Inference (all k-values) ----
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 2: vLLM inference (all k-values)" | tee -a "$LOG"

    for K in $K_VALUES; do
        TRIPLET_FILE="$BASE/k${K}/triplet-analysis/selected_triplets.json"
        RESULT_DIR="$BASE/k${K}/model-result"
        METRICS_FILE="$RESULT_DIR/llm_metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "  [k=$K] LLM results exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] Running vLLM inference (top-k=$K)..." | tee -a "$LOG"
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
    echo "K-ABLATION-COSINE COMPLETE. Results in: $BASE/"               | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"


# ============================================================================
# FULL-PIPELINE: Train → Coverage → Triplet Selection → vLLM Inference
# ============================================================================
# Usage: just full-pipeline cwq
#        just full-pipeline webqsp
#        just full-pipeline cwq 50       (override top-k, default from config.yml)
#        just full-pipeline cwq 50 500   (override top-k and sample-k)

full-pipeline dataset topk="" samplek="":
    #!/usr/bin/env bash
    set -e

    # --- Read paths from config.yml ---
    YAML_OUTPUT=$(python3 scripts/read_config.py "{{dataset}}")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read config.yml for dataset '{{dataset}}'"
        exit 1
    fi

    TRAIN=$(echo "$YAML_OUTPUT" | sed -n '1p')
    VAL=$(echo "$YAML_OUTPUT" | sed -n '2p')
    TEST=$(echo "$YAML_OUTPUT" | sed -n '3p')
    DEFAULT_TOPK=$(echo "$YAML_OUTPUT" | sed -n '6p')
    LLM_MODEL=$(echo "$YAML_OUTPUT" | sed -n '7p')

    # Use overrides if provided, otherwise use defaults from config
    TOPK="{{topk}}"
    if [ -z "$TOPK" ]; then
        TOPK="$DEFAULT_TOPK"
    fi

    SAMPLEK="{{samplek}}"
    if [ -z "$SAMPLEK" ]; then
        SAMPLEK="1000"
    fi

    BASE="./results/full-pipeline/{{dataset}}/k${TOPK}-N${SAMPLEK}"
    LOG="logs/full-pipeline.log"
    mkdir -p logs

    echo "============================================================" | tee -a "$LOG"
    echo "FULL PIPELINE: {{dataset}} | top-k=$TOPK | sample-k=$SAMPLEK" | tee -a "$LOG"
    echo "  Train: $TRAIN"                                              | tee -a "$LOG"
    echo "  Val:   $VAL"                                                | tee -a "$LOG"
    echo "  Test:  $TEST"                                               | tee -a "$LOG"
    echo "  LLM:   $LLM_MODEL"                                         | tee -a "$LOG"
    echo "  Output: $BASE/"                                             | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # ---- STEP 1: Train model ----
    MODEL_DIR="$BASE/model"
    CKPT="$MODEL_DIR/main_training_k${TOPK}/best_model_k${TOPK}.pt"

    if [ -f "$CKPT" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 1: Model exists at $CKPT. Skipping." | tee -a "$LOG"
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 1: Training (k=$TOPK, N=$SAMPLEK, epochs=30, patience=10)..." | tee -a "$LOG"
        python cli.py train \
            --train-data "$TRAIN" \
            --val-data "$VAL" \
            --checkpoint-dir "$MODEL_DIR" \
            --k $TOPK \
            --sample-k $SAMPLEK \
            --num-epochs 30 \
            --early-stopping-patience 10 \
            2>&1 | tee -a "$LOG"
    fi

    # ---- STEP 2: Triplet selection (generates selected_triplets.json) ----
    TRIPLET_DIR="$BASE/triplet-analysis"
    TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

    if [ -f "$TRIPLET_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 2: selected_triplets.json exists. Skipping." | tee -a "$LOG"
    elif [ ! -f "$CKPT" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 2: ERROR - Model checkpoint not found. Cannot generate triplets." | tee -a "$LOG"
        exit 1
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 2: Generating triplets (top-k=$TOPK, sample-k=$SAMPLEK)..." | tee -a "$LOG"
        python -m src.utils.triplet_selector \
            --model-path "$CKPT" \
            --test-data "$TEST" \
            --output-dir "$TRIPLET_DIR" \
            --top-k $TOPK \
            --sample-k $SAMPLEK \
            2>&1 | tee -a "$LOG"
    fi

    # ---- STEP 3: Coverage analysis (ans_present + path_coverage) ----
    COVERAGE_DIR="$BASE/coverage"
    COVERAGE_FILE="$COVERAGE_DIR/coverage_metrics.json"

    if [ -f "$COVERAGE_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: Coverage metrics exist. Skipping." | tee -a "$LOG"
    elif [ ! -f "$CKPT" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: ERROR - Model checkpoint not found. Cannot compute coverage." | tee -a "$LOG"
        exit 1
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: Computing coverage (ans_present, path_coverage)..." | tee -a "$LOG"
        python scripts/run_coverage.py "$CKPT" "$TEST" $TOPK "$COVERAGE_FILE" \
            2>&1 | tee -a "$LOG"
    fi

    # ---- STEP 4: vLLM LLM Inference (hit, hit@1, f1, precision, recall) ----
    LLM_DIR="$BASE/llm-inference"
    LLM_METRICS="$LLM_DIR/llm_metrics.json"

    if [ -f "$LLM_METRICS" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 4: LLM metrics exist. Skipping." | tee -a "$LOG"
    elif [ ! -f "$TRIPLET_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 4: ERROR - selected_triplets.json not found. Cannot run LLM." | tee -a "$LOG"
        exit 1
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 4: Running vLLM inference (top-k=$TOPK)..." | tee -a "$LOG"
        python run_vllm_inference_ablation.py \
            --input "$TRIPLET_FILE" \
            --output "$LLM_DIR" \
            --llm-model "$LLM_MODEL" \
            --top-k $TOPK \
            2>&1 | tee -a "$LOG"
    fi

    # ---- Summary ----
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "FULL PIPELINE COMPLETE: {{dataset}} | k=$TOPK | N=$SAMPLEK"   | tee -a "$LOG"
    echo "  Results: $BASE/"                                            | tee -a "$LOG"
    echo "    model/             - trained checkpoint"                   | tee -a "$LOG"
    echo "    triplet-analysis/  - selected_triplets.json"              | tee -a "$LOG"
    echo "    coverage/          - coverage_metrics.json"               | tee -a "$LOG"
    echo "    llm-inference/     - llm_metrics.json"                    | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
