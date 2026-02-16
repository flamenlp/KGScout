# KGScout

Official implementation of paper **KG-Scout: A Policy Driven Knowledge-Graph Retrieval to Mitigate Factual Inaccuracies of Large Language Model**

## Overview

KGScout is a two-phase KGQA pipeline that:
1. Uses a trained path ranking model to select top-k relevant triplets from a knowledge graph
2. Employs Large Language Models (LLMs) to generate answers from the selected triplets

Comprehensive evaluation analysis including LLM comparison, k-value ablation studies, coverage analysis, and statistical comparisons between retriever methods are done here

## Installation

```bash
pip install -r requirements.txt
```

## Quick Reference

### Pipeline Workflow

1. **Preprocess** → Prepare datasets with embeddings and graph features
2. **Train** → Train the KGscout path ranking model
3. **Inference** → Select top-k triplets using trained model
4. **Evaluate** → Compute coverage metrics
5. **LLM Analysis** → Compare LLMs, run ablation studies, analyze coverage and statistics

### Command Summary

| Command | Purpose |
|---------|---------|
| `train` | Train KGscout model|
| `inference` | Select top-k triplets from test data |
| `evaluate` | Compute answer/path coverage metrics |
| `llm-comparison` | Compare different LLM models |
| `k-ablation` | Evaluate different k values with Llama-3.1-8b |
| `coverage-analysis` | Analyze coverage across k values and retrievers |
| `statistical-analysis` | Statistical comparison with case categorization |

## CLI Commands

The system provides a unified CLI interface with multiple commands for different pipeline stages.

### Pipeline Commands

#### 1. Preprocess Dataset

Prepare datasets with PPR (Personalized PageRank) features.

```bash
python cli.py preprocess_dataset \
  --input <input_file.json> \
  --output <output_directory>
```

**Parameters:**
- `--input`: Path to input data file
- `--output`: Path to output directory for preprocessed data

**Input Format (JSON):**
```json
[
  {
    "question": "What year did the team with mascot Lou Seal win?",
    "q_entity": ["Lou Seal", "San Francisco Giants"],
    "a_entity": ["2010 World Series", "2012 World Series"],
    "answer": ["2010", "2012", "2014"],
    "triplets": [
      ["Lou Seal", "sports.mascot.team", "San Francisco Giants"],
      ["San Francisco Giants", "sports.team.championships", "2010 World Series"]
    ]
  }
]
```

**Output:**
- Preprocessed PyTorch tensor file (`.pt`) with embeddings and graph features
- Saved to `{output_directory}/preprocessed_data.pt`

---

#### 2. Train Model

Run complete training pipeline (pretraining → main training).

```bash
python cli.py train \
  --train-data <train_file.pt> \
  --val-data <val_file.pt> \
  --checkpoint-dir <checkpoint_directory> \
  --k <30|50|100|150> \
  [--num-epochs 50] \
  [--learning-rate 1e-4] \
  [--weight-decay 1e-5] \
  [--gradient-accumulation-steps 8] \
  [--validation-interval 1] \
  [--early-stopping-patience 10] \
  [--warmup-steps 100]
```

**Parameters:**
- `--train-data`: Path to training data file (preprocessed `.pt` file)
- `--val-data`: Path to validation data file (preprocessed `.pt` file)
- `--checkpoint-dir`: Directory to save model checkpoints
- `--k`: K value for main training phase (choices: 30, 50, 100, 150)
- `--num-epochs`: Number of epochs for main training (default: 50)
- `--learning-rate`: Learning rate for training (default: 1e-4)
- `--weight-decay`: Weight decay for optimizer (default: 1e-5)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 8)
- `--validation-interval`: Validate every N epochs (default: 1)
- `--early-stopping-patience`: Early stopping patience in epochs (default: 10)
- `--warmup-steps`: Warmup steps for learning rate scheduler (default: 100)

**Training Pipeline:**
1. **Pretraining Phase**: 5 epochs with n=500 (fixed)
2. **Main Training Phase**: Uses specified k parameter and hyperparameters

**Output:**
- Pretraining checkpoints: `{checkpoint_dir}/pretrain/`
- Main training checkpoints: `{checkpoint_dir}/main/`
- Training logs: `{checkpoint_dir}/logs/`
- Best model: `{checkpoint_dir}/main/path_ranker.pt`

**Example:**
```bash
python cli.py train \
  --train-data dataset/processed-dataset/webqsp-train.pt \
  --val-data dataset/processed-dataset/webqsp-val.pt \
  --checkpoint-dir checkpoints/webqsp-k100/ \
  --k 100 \
  --num-epochs 50 \
  --learning-rate 1e-4
```

---

#### 3. Run Inference

Run inference to select top-k triplets from test data.

```bash
python cli.py inference \
  --model-path <model_checkpoint> \
  --test-data <test_file.pt> \
  --output-dir <output_directory> \
  [--top-k 100]
```

**Parameters:**
- `--model-path`: Path to trained model checkpoint
- `--test-data`: Path to test data file (preprocessed `.pt` file)
- `--output-dir`: Directory to save inference results
- `--top-k`: Number of top triplets to select (default: 100)

**Output:**
- Selected triplets JSON: `{output_dir}/selected_triplets.json`
- Inference statistics: Average reward, total samples processed

**Example:**
```bash
python cli.py inference \
  --model-path checkpoints/best_model/ \
  --test-data dataset/processed-dataset/webqsp-tst.pt \
  --output-dir results/inference/ \
  --top-k 100
```

---

#### 4. Evaluate Model

Compute evaluation metrics (answer coverage, path coverage).

```bash
python cli.py evaluate \
  --model-path <model_checkpoint> \
  --test-data <test_file.pt> \
  [--top-k 100]
```

**Parameters:**
- `--model-path`: Path to trained model checkpoint
- `--test-data`: Path to test data file (preprocessed `.pt` file)
- `--top-k`: Number of top triplets to evaluate (default: 100)

**Output (printed to console):**
- Answer Coverage: Percentage of questions where answer entities exist in selected triplets
- Path Coverage: Percentage of questions where complete reasoning path exists
- Average Reward: Mean reward score across all questions

**Example:**
```bash
python cli.py evaluate \
  --model-path checkpoints/best_model/ \
  --test-data dataset/processed-dataset/webqsp-tst.pt \
  --top-k 100
```

---

### Evaluation Analysis Commands

#### 5. LLM Comparison Analysis

Compare different LLM models using the same retriever configuration.

```bash
python cli.py llm-comparison \
  --dataset <webqsp|cwq> \
  --llm-model <llama|qwen|deepseek> \
  --retriever-type <kgscout|cosine> \
  --k <number> \
  --model-path <path_to_model> \
  --output-dir <output_directory>
```

**Parameters:**
- `--dataset`: Dataset to evaluate (`webqsp` or `cwq`)
- `--llm-model`: LLM model to use (`llama` for Llama-3.1-8b, `qwen`, or `deepseek`)
- `--retriever-type`: Triplet selection method (`kgscout` or `cosine`)
- `--k`: Number of top triplets to select
- `--model-path`: Path to trained KGscout model (required if `--retriever-type kgscout`)
- `--output-dir`: Directory to save results (default: `results`)

**Input Requirements:**
- Preprocessed dataset at `dataset/processed-dataset/{dataset}-tst.pt`
- Trained model checkpoint (if using KGscout retriever)
- Dataset format: PyTorch tensor file with fields:
  - `question`: Question text
  - `q_entity`: Question entities
  - `a_entity`: Answer entities
  - `answer`: Ground truth answers
  - `question_embedding`: Question embedding (384-dim)
  - `topk_linearized_triplets`: Pre-computed cosine triplets
  - `topk_rel_data`: List of (score, (subject, relation, object)) tuples
  - `topK_rel_embeddings`: Relation embeddings
  - `graph_features`: Graph structure features

**Output:**
- Directory structure: `{output_dir}/{dataset}/{retriever}-k{k}/`
- Files generated:
  - `predictions.txt`: One prediction per line
  - `results.jsonl`: Detailed per-question results (JSON Lines format)
  - `summary.json`: Aggregated metrics
    - Hit, Hit@1, Macro F1, Precision, Recall, Exact Match
    - Total questions processed

**Example:**
```bash
# Compare Llama with KGscout retriever
python cli.py llm-comparison \
  --dataset webqsp \
  --llm-model llama \
  --retriever-type kgscout \
  --k 100 \
  --model-path checkpoints/best_model/ \
  --output-dir results/

# Compare Qwen with cosine retriever
python cli.py llm-comparison \
  --dataset cwq \
  --llm-model qwen \
  --retriever-type cosine \
  --k 50 \
  --output-dir results/
```

---

#### 6. K-Value Ablation Study

Evaluate how different k values affect answer quality using Llama-3.1-8b.

```bash
python cli.py k-ablation \
  --dataset <webqsp|cwq> \
  --retriever-type <kgscout|cosine> \
  --k-values <comma_separated_values> \
  --model-path <path_to_model> \
  --output-dir <output_directory>
```

**Parameters:**
- `--dataset`: Dataset to evaluate (`webqsp` or `cwq`)
- `--retriever-type`: Triplet selection method (`kgscout` or `cosine`)
- `--k-values`: Comma-separated list of k values (default: `30,50,100,150`)
- `--k`: Single k value (overrides `--k-values` if provided)
- `--model-path`: Path to trained KGscout model (required if `--retriever-type kgscout`)
- `--output-dir`: Directory to save results (default: `results`)

**Input Requirements:**
- Same as LLM Comparison Analysis

**Output:**
- Individual results for each k-value in: `{output_dir}/{dataset}/{retriever}-k{k}/`
- Summary file: `{output_dir}/{dataset}/{retriever}-ablation/k_ablation_summary_{timestamp}.json`
  - Metadata: timestamp, dataset, retriever type, k-values tested
  - Results by k-value: metrics for each k
  - Comparison table: formatted table showing metrics across all k-values

**Example:**
```bash
# Test multiple k-values with KGscout
python cli.py k-ablation \
  --dataset webqsp \
  --retriever-type kgscout \
  --k-values 30,50,100,150 \
  --model-path checkpoints/best_model/ \
  --output-dir results/

# Test single k-value with cosine
python cli.py k-ablation \
  --dataset cwq \
  --retriever-type cosine \
  --k 100 \
  --output-dir results/
```

---

#### 7. Coverage Analysis

Measure answer and path coverage for different k values, comparing KGscout vs Cosine retriever.

```bash
python cli.py coverage-analysis \
  --dataset <webqsp|cwq> \
  --model-path <path_to_model> \
  --k-values <comma_separated_values> \
  --output-dir <output_directory>
```

**Parameters:**
- `--dataset`: Dataset to evaluate (`webqsp` or `cwq`)
- `--model-path`: Path to trained KGscout model
- `--k-values`: Comma-separated list of k values (default: `30,50,100,150`)
- `--output-dir`: Directory to save results (default: `results/coverage`)

**Input Requirements:**
- Same as LLM Comparison Analysis

**Output:**
- File: `{output_dir}/coverage_analysis_{timestamp}.json`
  - Metadata: timestamp, dataset, k-values
  - Results: Coverage metrics for both retrievers at each k-value
    - `answer_coverage`: Percentage of questions where answer entities exist in triplets
    - `path_coverage`: Percentage of questions where complete reasoning path exists
    - Counts and totals for each metric
  - Comparison table: Formatted table comparing both retrievers

**Example:**
```bash
python cli.py coverage-analysis \
  --dataset webqsp \
  --model-path checkpoints/best_model/ \
  --k-values 30,50,100,150 \
  --output-dir results/coverage/
```

---

#### 8. Statistical Analysis

Comprehensive statistical comparison between Cosine and KGscout retrievers with case categorization.

```bash
python cli.py statistical-analysis \
  --dataset <webqsp|cwq> \
  --model-path <path_to_model> \
  --k <number> \
  --output-dir <output_directory>
```

**Parameters:**
- `--dataset`: Dataset to evaluate (`webqsp` or `cwq`)
- `--model-path`: Path to trained KGscout model
- `--k`: Number of top triplets to select
- `--output-dir`: Directory to save results (default: `results/statistical`)

**Input Requirements:**
- Same as LLM Comparison Analysis

**Output:**
- File: `{output_dir}/statistical_analysis_{timestamp}.json`
  - Metadata: timestamp, dataset, k, total questions
  - Case statistics: Count and percentage for each case
    - **Case 1**: Cosine no relevant, KGscout some relevant
    - **Case 2**: Cosine relevant no path, KGscout has path
    - **Case 3**: Both have relevant triplets (overlapping paths)
    - **Case 4**: Both have relevant triplets (non-overlapping paths)
    - **Case 5**: Cosine better than KGscout
    - **Case 6**: Both fail
  - Examples per case: Up to 5 example questions for each case

**Example:**
```bash
python cli.py statistical-analysis \
  --dataset webqsp \
  --model-path checkpoints/best_model/ \
  --k 100 \
  --output-dir results/statistical/
```

---

## Dataset Format

Preprocessed datasets should be PyTorch tensor files (`.pt`) with the following structure:

```python
[
  {
    "question": str,                    # Natural language question
    "q_entity": List[str],              # Question entities
    "a_entity": List[str],              # Answer entities
    "answer": List[str],                # Ground truth answers
    "question_embedding": torch.Tensor, # Question embedding (384-dim)
    "topk_linearized_triplets": List[str],  # Cosine-selected triplets
    "topk_linearized_triplet_embeddings": torch.Tensor,  # Triplet embeddings
    "topk_rel_data": List[Tuple[float, Tuple[str, str, str]]],  # (score, (s, r, o))
    "topK_rel_embeddings": torch.Tensor,  # Relation embeddings
    "graph_features": torch.Tensor,     # Graph structure features
    "is_empty": bool                    # Whether question has no valid triplets
  },
  ...
]
```

**Expected Locations:**
- WebQSP test set: `dataset/processed-dataset/webqsp-tst.pt`
- CWQ test set: `dataset/processed-dataset/cwq-tst.pt`

---

## Model Checkpoint Format

KGscout model checkpoints should be saved using the `PathRankingModel.save_pretrained()` method.

**Expected Structure:**
```
checkpoints/best_model/
├── path_ranker.pt          # Model state dict
└── config.json             # Model configuration (optional)
```

The checkpoint should contain:
- `model_state_dict`: Model parameters
- `hidden_size`: Hidden dimension size
- `temperature`: Temperature parameter
- `baseline`: Baseline value

---

## Output File Formats

### LLM Comparison & K-Ablation

**predictions.txt**
```
predicted_answer_1
predicted_answer_2
...
```

**results.jsonl** (JSON Lines format)
```json
{"question": "...", "predicted": ["..."], "ground_truth": ["..."], "hit": 1.0, "f1": 0.85, ...}
{"question": "...", "predicted": ["..."], "ground_truth": ["..."], "hit": 0.0, "f1": 0.0, ...}
```

**summary.json**
```json
{
  "dataset": "webqsp",
  "retriever_type": "kgscout",
  "k": 100,
  "metrics": {
    "hit": 0.75,
    "hit_at_1": 0.68,
    "macro_f1": 0.72,
    "macro_precision": 0.70,
    "macro_recall": 0.74,
    "exact_match": 0.45
  },
  "total_questions": 1000
}
```

### Coverage Analysis

**coverage_analysis_{timestamp}.json**
```json
{
  "metadata": {
    "timestamp": "20240315_143022",
    "dataset": "webqsp",
    "k_values": [30, 50, 100, 150]
  },
  "results": {
    "kgscout": {
      "30": {"answer_coverage": 0.75, "path_coverage": 0.60},
      "50": {"answer_coverage": 0.82, "path_coverage": 0.68}
    },
    "cosine": {
      "30": {"answer_coverage": 0.70, "path_coverage": 0.55},
      "50": {"answer_coverage": 0.78, "path_coverage": 0.63}
    }
  },
  "comparison_table": "..."
}
```

### Statistical Analysis

**statistical_analysis_{timestamp}.json**
```json
{
  "metadata": {
    "timestamp": "20240315_143022",
    "dataset": "webqsp",
    "k": 100,
    "total_questions": 1000
  },
  "case_statistics": {
    "case1": {
      "count": 150,
      "percentage": 15.0,
      "description": "Cosine no relevant, KGscout some relevant"
    },
    ...
  },
  "examples_per_case": {
    "case1": [
      {
        "question_id": 42,
        "question": "...",
        "cosine_triplet_count": 0,
        "kgscout_triplet_count": 100
      },
      ...
    ]
  }
}
```

---

## Metrics Explained

- **Hit**: Whether any predicted answer matches any ground truth answer (binary: 0 or 1)
- **Hit@1**: Whether the first predicted answer matches any ground truth answer
- **Macro F1**: Average F1 score across all questions
- **Macro Precision**: Average precision across all questions
- **Macro Recall**: Average recall across all questions
- **Exact Match**: Whether all predictions match all ground truths exactly
- **Answer Coverage**: Percentage of questions where answer entities exist in selected triplets
- **Path Coverage**: Percentage of questions where a complete reasoning path exists from question entities to answer entities

---

## Error Handling

All commands include comprehensive error handling:

- **Missing files**: Clear error messages with expected file paths
- **Invalid parameters**: Validation with helpful suggestions
- **Processing failures**: Individual question failures are logged but don't stop execution
- **Output errors**: Directory creation and file write validation

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers (for LLM inference)
- tqdm (for progress bars)
- NetworkX (for path coverage analysis)

See `requirements.txt` for complete dependencies.