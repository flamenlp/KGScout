# KGScout Experiment Runner
# All paths are read from config.yml
#
# Usage: just k-ablation cwq
#        just k-ablation webqsp
#        just full-pipeline metaqa          (uses default 2-hop)
#        just full-pipeline metaqa 100 1000 (override top-k and sample-k)

# Default MetaQA hop (used when dataset=metaqa)
metaqa_dataset_hop := "2"

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
        TRAIN_DIR="$MODEL_DIR/main_training_k${K}"

        # Find best checkpoint using the helper script
        CKPT=""
        if [ -d "$TRAIN_DIR" ]; then
            CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
        fi

        if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
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
            # Re-find checkpoint after training
            CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
        fi

        # ---- STEP 2: Triplet selection ----
        TRIPLET_DIR="$BASE/k${K}/triplet-analysis"
        TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

        if [ -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] STEP 2: selected_triplets.json exists. Skipping." | tee -a "$LOG"
        elif [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [k=$K] STEP 2: ERROR: Model checkpoint not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] STEP 2: Generating triplets (top-k=$K)..." | tee -a "$LOG"
            python -m src.utils.triplet_selector \
                --model-path "$CKPT" \
                --test-data "$TEST" \
                --output-dir "$TRIPLET_DIR" \
                --top-k $K \
                2>&1 | tee -a "$LOG"
        fi

        # ---- STEP 3: Coverage analysis (from selected_triplets.json) ----
        COVERAGE_DIR="$BASE/k${K}/triplet_metrics"
        COVERAGE_FILE="$COVERAGE_DIR/coverage_metrics.json"

        if [ -f "$COVERAGE_FILE" ]; then
            echo "  [k=$K] STEP 3: Coverage metrics exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] STEP 3: ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] STEP 3: Computing coverage (ans_present, path_coverage)..." | tee -a "$LOG"
            python scripts/run_coverage_from_triplets.py \
                "$TRIPLET_FILE" "$COVERAGE_FILE" \
                2>&1 | tee -a "$LOG"
        fi

        # ---- STEP 4: vLLM LLM Inference ----
        RESULT_DIR="$BASE/k${K}/model-result"
        METRICS_FILE="$RESULT_DIR/llm_metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "  [k=$K] STEP 4: LLM results exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] STEP 4: ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] STEP 4: Running vLLM inference (top-k=$K)..." | tee -a "$LOG"
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

k-ablation-cosine dataset llm="":
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

    # Override LLM model if provided
    LLM_OVERRIDE="{{llm}}"
    if [ -n "$LLM_OVERRIDE" ]; then
        LLM_MODEL="$LLM_OVERRIDE"
    fi

    # Cosine ablation base dir with dataset subdirectory
    BASE="./results/cosine-k-ablation/{{dataset}}"

    LOG="logs/cosine-k-ablation.log"
    mkdir -p logs

    echo "============================================================" | tee -a "$LOG"
    echo "K-ABLATION-COSINE: {{dataset}} | k=$K_VALUES | llm=$LLM_MODEL" | tee -a "$LOG"
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

    # ---- PHASE 2: Coverage analysis (all k-values) ----
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 2: Coverage analysis (all k-values)" | tee -a "$LOG"

    for K in $K_VALUES; do
        TRIPLET_FILE="$BASE/k${K}/triplet-analysis/selected_triplets.json"
        COVERAGE_DIR="$BASE/k${K}/triplet_metrics"
        COVERAGE_FILE="$COVERAGE_DIR/coverage_metrics.json"

        if [ -f "$COVERAGE_FILE" ]; then
            echo "  [k=$K] Coverage metrics exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] ERROR: selected_triplets.json not found. Skipping coverage." | tee -a "$LOG"
        else
            echo "  [k=$K] Computing coverage (ans_present, path_coverage)..." | tee -a "$LOG"
            python scripts/run_coverage_from_triplets.py \
                "$TRIPLET_FILE" "$COVERAGE_FILE" \
                2>&1 | tee -a "$LOG"
        fi
    done

    # ---- PHASE 3: vLLM LLM Inference (all k-values) ----
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 3: vLLM inference (all k-values, llm=$LLM_MODEL)" | tee -a "$LOG"

    for K in $K_VALUES; do
        TRIPLET_FILE="$BASE/k${K}/triplet-analysis/selected_triplets.json"
        RESULT_DIR="$BASE/k${K}/${LLM_MODEL}-inference"
        METRICS_FILE="$RESULT_DIR/llm_metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "  [k=$K] LLM results exist ($LLM_MODEL). Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [k=$K] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [k=$K] Running vLLM inference (top-k=$K, llm=$LLM_MODEL)..." | tee -a "$LOG"
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
    echo "K-ABLATION-COSINE COMPLETE. Results in: $BASE/ (llm=$LLM_MODEL)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"


# ============================================================================
# PREPROCESS-METAQA: Download & preprocess MetaQA data for training
# ============================================================================
# Usage: just preprocess-metaqa          (default: 2-hop)
#        just preprocess-metaqa 3        (override hop)

preprocess-metaqa hop=metaqa_dataset_hop:
    #!/usr/bin/env bash

    echo "============================================================"
    echo "PREPROCESS MetaQA {{hop}}-hop"
    echo "============================================================"

    # Read paths from config.yml using helper script
    YAML_OUT=$(python3 scripts/read_metaqa_config.py "{{hop}}")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read config.yml for metaqa {{hop}}-hop"
        exit 1
    fi

    KB_PATH=$(echo "$YAML_OUT" | sed -n '1p')
    QA_TRAIN=$(echo "$YAML_OUT" | sed -n '2p')
    QA_DEV=$(echo "$YAML_OUT" | sed -n '3p')
    QA_TEST=$(echo "$YAML_OUT" | sed -n '4p')
    OUTPUT_DIR=$(echo "$YAML_OUT" | sed -n '5p')
    EMBED_MODEL=$(echo "$YAML_OUT" | sed -n '6p')

    # Check if raw data exists
    if [ ! -f "$KB_PATH" ]; then
        echo "ERROR: kb.txt not found at: $KB_PATH"
        echo "Please download MetaQA dataset first."
        echo "  See: https://github.com/yuyuz/MetaQA"
        echo "  Place files in data/metaqa/ with structure:"
        echo "    data/metaqa/kb.txt"
        echo "    data/metaqa/{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt"
        exit 1
    fi

    # Check if already preprocessed
    TRAIN_FILE="$OUTPUT_DIR/metaqa-{{hop}}hop-train.pt"
    if [ -f "$TRAIN_FILE" ]; then
        echo "Preprocessed data already exists: $TRAIN_FILE"
        echo "Delete it to re-run preprocessing."
        exit 0
    fi

    echo "  KB:       $KB_PATH"
    echo "  Train:    $QA_TRAIN"
    echo "  Dev:      $QA_DEV"
    echo "  Test:     $QA_TEST"
    echo "  Output:   $OUTPUT_DIR"
    echo "  Embed:    $EMBED_MODEL"
    echo "============================================================"

    python generalization-study/preprocess_metaqa.py \
        --kb-path "$KB_PATH" \
        --qa-train-path "$QA_TRAIN" \
        --qa-dev-path "$QA_DEV" \
        --qa-test-path "$QA_TEST" \
        --output-dir "$OUTPUT_DIR" \
        --hop {{hop}} \
        --embedding-model "$EMBED_MODEL"

    echo ""
    echo "============================================================"
    echo "PREPROCESSING COMPLETE"
    echo "============================================================"


# ============================================================================
# FULL-PIPELINE: Train → Coverage → Triplet Selection → vLLM Inference
# ============================================================================
# Usage: just full-pipeline cwq
#        just full-pipeline webqsp
#        just full-pipeline metaqa          (uses default 2-hop from config)
#        just full-pipeline cwq 50       (override top-k, default from config.yml)
#        just full-pipeline cwq 50 500   (override top-k and sample-k)
#        just full-pipeline cwq 100 1000 qwen  (override LLM model)

full-pipeline dataset topk="" samplek="" llm="":
    #!/usr/bin/env bash

    # --- Read paths from config.yml ---
    # For metaqa, pass the hop argument to resolve hop-specific paths
    if [ "{{dataset}}" = "metaqa" ]; then
        YAML_OUTPUT=$(python3 scripts/read_config.py "{{dataset}}" "{{metaqa_dataset_hop}}")
    else
        YAML_OUTPUT=$(python3 scripts/read_config.py "{{dataset}}")
    fi

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

    # Override LLM model if provided
    LLM_OVERRIDE="{{llm}}"
    if [ -n "$LLM_OVERRIDE" ]; then
        LLM_MODEL="$LLM_OVERRIDE"
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
    TRAIN_DIR="$MODEL_DIR/main_training_k${TOPK}"

    # Find best checkpoint using the helper script
    CKPT=""
    if [ -d "$TRAIN_DIR" ]; then
        CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
    fi

    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
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
        # Re-find checkpoint after training
        CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
    fi

    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo ">>> ERROR: No best checkpoint found after training." | tee -a "$LOG"
        exit 1
    fi
    # CKPT_DIR is the directory containing path_ranker.pt (needed for from_pretrained)
    CKPT_DIR=$(dirname "$CKPT")

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

    # ---- STEP 3: Coverage analysis (from selected_triplets.json) ----
    COVERAGE_DIR="$BASE/triplet_metrics"
    COVERAGE_FILE="$COVERAGE_DIR/coverage_metrics.json"

    if [ -f "$COVERAGE_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: Coverage metrics exist. Skipping." | tee -a "$LOG"
    elif [ ! -f "$TRIPLET_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: ERROR - selected_triplets.json not found. Cannot compute coverage." | tee -a "$LOG"
        exit 1
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: Computing coverage (ans_present, path_coverage)..." | tee -a "$LOG"
        python scripts/run_coverage_from_triplets.py \
            "$TRIPLET_FILE" "$COVERAGE_FILE" \
            2>&1 | tee -a "$LOG"
    fi

    # ---- STEP 4: vLLM LLM Inference (hit, hit@1, f1, precision, recall) ----
    LLM_DIR="$BASE/${LLM_MODEL}-inference"
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
        echo ">>> STEP 4: Running vLLM inference (top-k=$TOPK, llm=$LLM_MODEL)..." | tee -a "$LOG"
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
    echo "FULL PIPELINE COMPLETE: {{dataset}} | k=$TOPK | N=$SAMPLEK | llm=$LLM_MODEL" | tee -a "$LOG"
    echo "  Results: $BASE/"                                            | tee -a "$LOG"
    echo "    model/             - trained checkpoint"                   | tee -a "$LOG"
    echo "    triplet-analysis/  - selected_triplets.json"              | tee -a "$LOG"
    echo "    triplet_metrics/   - coverage_metrics.json"               | tee -a "$LOG"
    echo "    ${LLM_MODEL}-inference/ - llm_metrics.json"               | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"


# ============================================================================
# ABLATION-2: Model Architecture + Reward Function Ablation Studies
# ============================================================================
# Reversed attention ablations: 6 model variants + 2 reward variants.
# Pipeline per variant: Train → Triplet Selection → Coverage → vLLM Inference
#
# Training uses the same Pretrainer and Trainer from src/ (same as full-pipeline).
# Triplet selection is deterministic top-k (no sampling at inference).
# Hyperparameters match full-pipeline (gradient_accumulation_steps=32, etc.)
#
# Usage: just run-ablations cwq
#        just run-ablations webqsp

run-ablations dataset:
    #!/usr/bin/env bash

    # --- Read ablation config ---
    YAML_OUTPUT=$(python3 scripts/read_ablation_config.py "{{dataset}}")

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read config.yml for dataset '{{dataset}}'"
        exit 1
    fi

    TRAIN=$(echo "$YAML_OUTPUT" | sed -n '1p')
    VAL=$(echo "$YAML_OUTPUT" | sed -n '2p')
    TEST=$(echo "$YAML_OUTPUT" | sed -n '3p')
    DEFAULT_TOPK=$(echo "$YAML_OUTPUT" | sed -n '4p')
    LLM_MODEL=$(echo "$YAML_OUTPUT" | sed -n '5p')
    MODEL_VARIANTS=$(echo "$YAML_OUTPUT" | sed -n '6p')
    REWARD_VARIANTS=$(echo "$YAML_OUTPUT" | sed -n '7p')
    MODEL_BASE=$(echo "$YAML_OUTPUT" | sed -n '8p')
    REWARD_BASE=$(echo "$YAML_OUTPUT" | sed -n '9p')
    NUM_EPOCHS=$(echo "$YAML_OUTPUT" | sed -n '10p')
    PATIENCE=$(echo "$YAML_OUTPUT" | sed -n '11p')

    LOG="logs/ablation-2.log"
    mkdir -p logs

    echo "============================================================" | tee -a "$LOG"
    echo "ABLATION: {{dataset}}"                                        | tee -a "$LOG"
    echo "  Model variants:  $MODEL_VARIANTS"                           | tee -a "$LOG"
    echo "  Reward variants: $REWARD_VARIANTS"                          | tee -a "$LOG"
    echo "  Train: $TRAIN"                                              | tee -a "$LOG"
    echo "  Val:   $VAL"                                                | tee -a "$LOG"
    echo "  Test:  $TEST"                                               | tee -a "$LOG"
    echo "  LLM:   $LLM_MODEL"                                         | tee -a "$LOG"
    echo "  Top-k: $DEFAULT_TOPK"                                       | tee -a "$LOG"
    echo "  Epochs: $NUM_EPOCHS, Patience: $PATIENCE"                   | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # ================================================================
    # PHASE 1: Model Ablation — Train + Triplet Selection
    # ================================================================
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 1: Model Ablation — Train + Triplet Selection" | tee -a "$LOG"

    for V in $MODEL_VARIANTS; do
        CKPT_DIR="$MODEL_BASE/$V/model"
        TRAIN_DIR="$CKPT_DIR/main_training_k${DEFAULT_TOPK}"
        TRIPLET_DIR="$MODEL_BASE/$V/triplet-result"
        TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

        echo "" | tee -a "$LOG"
        echo "  [$V] -------------------------------------------" | tee -a "$LOG"

        if [ -f "$TRIPLET_FILE" ]; then
            echo "  [$V] selected_triplets.json exists. Skipping train+inference." | tee -a "$LOG"
            continue
        fi

        # Check if checkpoint exists
        CKPT=""
        if [ -d "$TRAIN_DIR" ]; then
            CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
        fi

        if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
            echo "  [$V] Checkpoint found at $CKPT. Skipping training." | tee -a "$LOG"
        else
            echo "  [$V] Training with model-class=$V..." | tee -a "$LOG"
            python cli.py train \
                --train-data "$TRAIN" \
                --val-data "$VAL" \
                --checkpoint-dir "$CKPT_DIR" \
                --k $DEFAULT_TOPK \
                --num-epochs $NUM_EPOCHS \
                --early-stopping-patience $PATIENCE \
                --model-class $V \
                2>&1 | tee -a "$LOG"
            # Find checkpoint after training
            CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
        fi

        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [$V] ERROR: No checkpoint found after training. Skipping inference." | tee -a "$LOG"
            continue
        fi

        echo "  [$V] Generating triplets (deterministic top-k=$DEFAULT_TOPK)..." | tee -a "$LOG"
        python -m src.utils.triplet_selector \
            --model-path "$CKPT" \
            --test-data "$TEST" \
            --output-dir "$TRIPLET_DIR" \
            --top-k $DEFAULT_TOPK \
            --model-class $V \
            2>&1 | tee -a "$LOG"
    done

    # ================================================================
    # PHASE 2: Reward Ablation — Train + Triplet Selection
    # ================================================================
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 2: Reward Ablation — Train + Triplet Selection" | tee -a "$LOG"

    for V in $REWARD_VARIANTS; do
        CKPT_DIR="$REWARD_BASE/$V/model"
        TRAIN_DIR="$CKPT_DIR/main_training_k${DEFAULT_TOPK}"
        TRIPLET_DIR="$REWARD_BASE/$V/triplet-result"
        TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

        echo "" | tee -a "$LOG"
        echo "  [$V] -------------------------------------------" | tee -a "$LOG"

        if [ -f "$TRIPLET_FILE" ]; then
            echo "  [$V] selected_triplets.json exists. Skipping train+inference." | tee -a "$LOG"
            continue
        fi

        # Check if checkpoint exists
        CKPT=""
        if [ -d "$TRAIN_DIR" ]; then
            CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
        fi

        if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
            echo "  [$V] Checkpoint found at $CKPT. Skipping training." | tee -a "$LOG"
        else
            echo "  [$V] Training with reward-function=$V..." | tee -a "$LOG"
            python cli.py train \
                --train-data "$TRAIN" \
                --val-data "$VAL" \
                --checkpoint-dir "$CKPT_DIR" \
                --k $DEFAULT_TOPK \
                --num-epochs $NUM_EPOCHS \
                --early-stopping-patience $PATIENCE \
                --reward-function $V \
                2>&1 | tee -a "$LOG"
            # Find checkpoint after training
            CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)
        fi

        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "  [$V] ERROR: No checkpoint found after training. Skipping inference." | tee -a "$LOG"
            continue
        fi

        # Reward ablation uses default PathRankingModel (no --model-class)
        echo "  [$V] Generating triplets (deterministic top-k=$DEFAULT_TOPK)..." | tee -a "$LOG"
        python -m src.utils.triplet_selector \
            --model-path "$CKPT" \
            --test-data "$TEST" \
            --output-dir "$TRIPLET_DIR" \
            --top-k $DEFAULT_TOPK \
            2>&1 | tee -a "$LOG"
    done

    # ================================================================
    # PHASE 3: Coverage Analysis (all variants)
    # ================================================================
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 3: Coverage Analysis" | tee -a "$LOG"

    # Model ablation coverage
    for V in $MODEL_VARIANTS; do
        TRIPLET_FILE="$MODEL_BASE/$V/triplet-result/selected_triplets.json"
        COVERAGE_FILE="$MODEL_BASE/$V/triplet_metrics/coverage_metrics.json"

        if [ -f "$COVERAGE_FILE" ]; then
            echo "  [model/$V] Coverage exists. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [model/$V] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [model/$V] Computing coverage..." | tee -a "$LOG"
            python scripts/run_coverage_from_triplets.py \
                "$TRIPLET_FILE" "$COVERAGE_FILE" \
                2>&1 | tee -a "$LOG"
        fi
    done

    # Reward ablation coverage
    for V in $REWARD_VARIANTS; do
        TRIPLET_FILE="$REWARD_BASE/$V/triplet-result/selected_triplets.json"
        COVERAGE_FILE="$REWARD_BASE/$V/triplet_metrics/coverage_metrics.json"

        if [ -f "$COVERAGE_FILE" ]; then
            echo "  [reward/$V] Coverage exists. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [reward/$V] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [reward/$V] Computing coverage..." | tee -a "$LOG"
            python scripts/run_coverage_from_triplets.py \
                "$TRIPLET_FILE" "$COVERAGE_FILE" \
                2>&1 | tee -a "$LOG"
        fi
    done

    # ================================================================
    # PHASE 4: vLLM Inference (all variants)
    # ================================================================
    echo "" | tee -a "$LOG"
    echo ">>> PHASE 4: vLLM Inference" | tee -a "$LOG"

    # Model ablation inference
    for V in $MODEL_VARIANTS; do
        TRIPLET_FILE="$MODEL_BASE/$V/triplet-result/selected_triplets.json"
        LLM_DIR="$MODEL_BASE/$V/llama-inference"
        LLM_METRICS="$LLM_DIR/llm_metrics.json"

        if [ -f "$LLM_METRICS" ]; then
            echo "  [model/$V] LLM metrics exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [model/$V] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [model/$V] Running vLLM inference..." | tee -a "$LOG"
            python run_vllm_inference_ablation.py \
                --input "$TRIPLET_FILE" \
                --output "$LLM_DIR" \
                --llm-model "$LLM_MODEL" \
                --top-k $DEFAULT_TOPK \
                2>&1 | tee -a "$LOG"
        fi
    done

    # Reward ablation inference
    for V in $REWARD_VARIANTS; do
        TRIPLET_FILE="$REWARD_BASE/$V/triplet-result/selected_triplets.json"
        LLM_DIR="$REWARD_BASE/$V/llama-inference"
        LLM_METRICS="$LLM_DIR/llm_metrics.json"

        if [ -f "$LLM_METRICS" ]; then
            echo "  [reward/$V] LLM metrics exist. Skipping." | tee -a "$LOG"
        elif [ ! -f "$TRIPLET_FILE" ]; then
            echo "  [reward/$V] ERROR: selected_triplets.json not found. Skipping." | tee -a "$LOG"
        else
            echo "  [reward/$V] Running vLLM inference..." | tee -a "$LOG"
            python run_vllm_inference_ablation.py \
                --input "$TRIPLET_FILE" \
                --output "$LLM_DIR" \
                --llm-model "$LLM_MODEL" \
                --top-k $DEFAULT_TOPK \
                2>&1 | tee -a "$LOG"
        fi
    done

    # ---- Summary ----
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "ABLATION COMPLETE: {{dataset}}"                               | tee -a "$LOG"
    echo "  Model ablation: $MODEL_BASE/"                               | tee -a "$LOG"
    echo "  Reward ablation: $REWARD_BASE/"                             | tee -a "$LOG"
    echo "  Per variant:"                                               | tee -a "$LOG"
    echo "    model/             - trained checkpoint"                   | tee -a "$LOG"
    echo "    triplet-result/    - selected_triplets.json"              | tee -a "$LOG"
    echo "    triplet_metrics/   - coverage_metrics.json"               | tee -a "$LOG"
    echo "    llama-inference/   - llm_metrics.json"                    | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"


# ============================================================================
# STATISTICAL-ANALYSIS: Compare cosine vs KGScout retrievers with case categorization
# ============================================================================
# Loads test dataset + model checkpoints directly (no JSON triplet parsing).
# Categorizes each question into 6 cases based on answer coverage and path overlap.
#
# Model checkpoint resolution:
#   k-ablation/{dataset}/k{K}/model/main_training_k{K}/checkpoint_best_epoch_*/path_ranker.pt
#   → fallback: full-pipeline/{dataset}/k{K}-N1000/model/main_training_k{K}/...
#
# Test data path: read from config.yml (datasets.{dataset}.test)
#
# Usage: just statistical-analysis cwq
#        just statistical-analysis webqsp
#        just statistical-analysis cwq "30 50 100 150"

# ============================================================================
# CROSS-DOMAIN: Test generalisation of a model trained on one dataset against
#               the test set of another dataset (zero-shot transfer).
# ============================================================================
# Loads the best k=100 checkpoint from full-pipeline/{src}/k100-N1000/,
# runs triplet selection on the target test set, computes retrieval metrics
# (answer_coverage, path_coverage) and LLM metrics (hit, hit@1, F1, EM)
# using the llama model.
#
# Checkpoint path:
#   full-pipeline/{src}/k100-N1000/model/main_training_k100/checkpoint_best_epoch_*/path_ranker.pt
#
# Output:
#   results/crossdomain/src-{src}-target-{tgt}/
#     triplet-result/selected_triplets.json
#     triplet_metrics/coverage_metrics.json
#     llama-inference/llm_metrics.json
#
# Usage: just cross-domain cwq webqsp
#        just cross-domain webqsp cwq

cross-domain src tgt:
    #!/usr/bin/env bash

    K=100
    SAMPLE_K=1000
    LLM_MODEL="llama"

    OUT_BASE="./results/crossdomain/src-{{src}}-target-{{tgt}}"
    LOG="logs/cross-domain.log"
    mkdir -p logs "$OUT_BASE"

    echo "============================================================" | tee -a "$LOG"
    echo "CROSS-DOMAIN: src={{src}} → target={{tgt}} | k=$K"           | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # ---- Read target test path from config.yml ----
    TGT_CONFIG=$(python3 scripts/read_config.py "{{tgt}}")
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read config.yml for target dataset '{{tgt}}'" | tee -a "$LOG"
        exit 1
    fi
    TGT_TEST=$(echo "$TGT_CONFIG" | sed -n '3p')

    if [ ! -f "$TGT_TEST" ]; then
        echo "ERROR: Target test file not found: $TGT_TEST" | tee -a "$LOG"
        exit 1
    fi
    echo "  Source:      {{src}}"      | tee -a "$LOG"
    echo "  Target:      {{tgt}}"      | tee -a "$LOG"
    echo "  Target test: $TGT_TEST"    | tee -a "$LOG"
    echo "  Output:      $OUT_BASE/"   | tee -a "$LOG"

    # ---- STEP 1: Resolve best checkpoint from full-pipeline/{src}/k100-N1000/ ----
    TRAIN_DIR="./results/full-pipeline/{{src}}/k${K}-N${SAMPLE_K}/model/main_training_k${K}"
    CKPT=$(python3 scripts/find_checkpoint.py "$TRAIN_DIR" 2>/dev/null)

    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "ERROR: No checkpoint found in $TRAIN_DIR" | tee -a "$LOG"
        echo "  Run: just full-pipeline {{src}} to train the source model first." | tee -a "$LOG"
        exit 1
    fi
    echo "  Checkpoint:  $CKPT" | tee -a "$LOG"

    # ---- STEP 2: Triplet selection ----
    TRIPLET_DIR="$OUT_BASE/triplet-result"
    TRIPLET_FILE="$TRIPLET_DIR/selected_triplets.json"

    if [ -f "$TRIPLET_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 2: selected_triplets.json exists. Skipping." | tee -a "$LOG"
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 2: Generating triplets (src={{src}}, target={{tgt}}, k=$K)..." | tee -a "$LOG"
        python -m src.utils.triplet_selector \
            --model-path "$CKPT" \
            --test-data "$TGT_TEST" \
            --output-dir "$TRIPLET_DIR" \
            --top-k $K \
            --sample-k $SAMPLE_K \
            2>&1 | tee -a "$LOG"
    fi

    if [ ! -f "$TRIPLET_FILE" ]; then
        echo "ERROR: selected_triplets.json not found after step 2. Aborting." | tee -a "$LOG"
        exit 1
    fi

    # ---- STEP 3: Retrieval metrics (answer_coverage, path_coverage) ----
    COVERAGE_DIR="$OUT_BASE/triplet_metrics"
    COVERAGE_FILE="$COVERAGE_DIR/coverage_metrics.json"

    if [ -f "$COVERAGE_FILE" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: Coverage metrics exist. Skipping." | tee -a "$LOG"
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 3: Computing retrieval metrics..." | tee -a "$LOG"
        python scripts/run_coverage_from_triplets.py \
            "$TRIPLET_FILE" "$COVERAGE_FILE" \
            2>&1 | tee -a "$LOG"
    fi

    # ---- STEP 4: LLM inference (llama, hit / hit@1 / F1 / EM) ----
    LLM_DIR="$OUT_BASE/llama-inference"
    LLM_METRICS="$LLM_DIR/llm_metrics.json"

    if [ -f "$LLM_METRICS" ]; then
        echo "" | tee -a "$LOG"
        echo ">>> STEP 4: LLM metrics exist. Skipping." | tee -a "$LOG"
    else
        echo "" | tee -a "$LOG"
        echo ">>> STEP 4: Running vLLM inference (llama, k=$K)..." | tee -a "$LOG"
        python run_vllm_inference_ablation.py \
            --input "$TRIPLET_FILE" \
            --output "$LLM_DIR" \
            --llm-model "$LLM_MODEL" \
            --top-k $K \
            2>&1 | tee -a "$LOG"
    fi

    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "CROSS-DOMAIN COMPLETE: src={{src}} → target={{tgt}}"          | tee -a "$LOG"
    echo "  Results: $OUT_BASE/"                                        | tee -a "$LOG"
    echo "    triplet-result/    - selected_triplets.json"              | tee -a "$LOG"
    echo "    triplet_metrics/   - coverage_metrics.json"               | tee -a "$LOG"
    echo "    llama-inference/   - llm_metrics.json"                    | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"


# ============================================================================
# HOP-ANALYSIS: Hop-stratified retriever analysis (KGScout vs cosine)
# ============================================================================
# Classifies test questions by reasoning hop count (1-hop, 2-hop, ≥3-hop,
# no-path) using the cosine top-1000 pool graph, then computes ans_present
# and has_path for both retrievers at each k value.
#
# Checkpoint resolution:
#   k-ablation/{dataset}/k{K}/model/main_training_k{K}/...  (k=30, 50, 150)
#   full-pipeline/{dataset}/k{K}-N1000/model/...            (k=100 fallback)
#
# Usage: just hop-analysis cwq
#        just hop-analysis webqsp
#        just hop-analysis cwq "30 50 100 150"

hop-analysis dataset kvalues="30 50 100 150":
    #!/usr/bin/env bash

    echo "============================================================"
    echo "HOP ANALYSIS: {{dataset}}"
    echo "  K values: {{kvalues}}"
    echo "============================================================"

    LOG="logs/hop-analysis.log"
    mkdir -p logs

    # Build --k-values argument
    K_ARGS=""
    for K in {{kvalues}}; do
        K_ARGS="$K_ARGS $K"
    done

    python scripts/run_hop_analysis.py \
        --dataset "{{dataset}}" \
        --k-values $K_ARGS \
        --results-base "./results" \
        --sample-k 1000 \
        2>&1 | tee -a "$LOG"

    echo ""
    echo "============================================================"
    echo "HOP ANALYSIS COMPLETE: {{dataset}}"
    echo "  Results: ./results/hop-analysis/{{dataset}}/"
    echo "============================================================"


# ============================================================================
# STATISTICAL-ANALYSIS: Compare cosine vs KGScout retrievers with case categorization
# ============================================================================
# Loads test dataset + model checkpoints directly (no JSON triplet parsing).
# Categorizes each question into 6 cases based on answer coverage and path overlap.
#
# Model checkpoint resolution:
#   k-ablation/{dataset}/k{K}/model/main_training_k{K}/checkpoint_best_epoch_*/path_ranker.pt
#   → fallback: full-pipeline/{dataset}/k{K}-N1000/model/main_training_k{K}/...
#
# Test data path: read from config.yml (datasets.{dataset}.test)
#
# Usage: just statistical-analysis cwq
#        just statistical-analysis webqsp
#        just statistical-analysis cwq "30 50 100 150"

statistical-analysis dataset kvalues="30 50 100 150":
    #!/usr/bin/env bash

    echo "============================================================"
    echo "STATISTICAL ANALYSIS: {{dataset}}"
    echo "  K values: {{kvalues}}"
    echo "============================================================"

    # Build --k-values argument
    K_ARGS=""
    for K in {{kvalues}}; do
        K_ARGS="$K_ARGS $K"
    done

    python scripts/run_statistical_analysis.py \
        --dataset "{{dataset}}" \
        --k-values $K_ARGS \
        --results-base "./results" \
        --sample-k 1000

    echo ""
    echo "============================================================"
    echo "STATISTICAL ANALYSIS COMPLETE: {{dataset}}"
    echo "  Results: ./results/statistical-analysis/{{dataset}}/"
    echo "============================================================"
