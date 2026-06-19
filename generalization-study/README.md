# MetaQA Generalization Study

Evaluates KGScout models (trained on WebQSP/CWQ over Freebase) on the MetaQA dataset
(movie domain KG) to test cross-KG generalizability.

## Setup

1. Download MetaQA dataset from: https://github.com/yuyuz/MetaQA
2. Extract to `data/metaqa/` so you have:
   ```
   data/metaqa/
   ├── kb.txt                    # Full movie knowledge graph
   ├── 1-hop/vanilla/
   │   ├── qa_train.txt
   │   ├── qa_dev.txt
   │   └── qa_test.txt
   ├── 2-hop/vanilla/
   │   ├── qa_train.txt
   │   ├── qa_dev.txt
   │   └── qa_test.txt
   └── 3-hop/vanilla/
       ├── qa_train.txt
       ├── qa_dev.txt
       └── qa_test.txt
   ```

## Pipeline

```bash
# Step 1: Preprocess MetaQA into KGScout-compatible format
python generalization-study/preprocess_metaqa.py \
    --kb-path data/metaqa/kb.txt \
    --qa-path data/metaqa/1-hop/vanilla/qa_test.txt \
    --output-dir data/metaqa/processed/ \
    --hop 1 \
    --max-triplets 1000

# Step 2: Run evaluation (model inference + LLM + metrics)
python generalization-study/run_generalization.py \
    --model-path checkpoints/webqsp-k100/main/ \
    --dataset-name webqsp \
    --hop 1 \
    --output-dir results/generalization/

# Run all hops at once:
python generalization-study/run_generalization.py \
    --model-path checkpoints/webqsp-k100/main/ \
    --dataset-name webqsp \
    --all-hops \
    --output-dir results/generalization/
```

## Test Set Sizes

| Hop | Test Questions |
|-----|---------------|
| 1   | 9,947         |
| 2   | 14,872        |
| 3   | 14,274        |

## Notes

- MetaQA uses a small movie KG (~135K triplets, 43K entities, 9 relations)
- KGScout was trained on Freebase-based datasets (WebQSP, CWQ) with thousands of relations
- This tests whether the model's learned ranking generalizes across KGs with different schemas
