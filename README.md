# KGScout

Official implementation of **KG-Scout: A Policy Driven Knowledge-Graph Retrieval to Mitigate Factual Inaccuracies of Large Language Model**.

## Overview

KGScout is a two-phase KGQA pipeline:
1. **Retrieval**: A trained path ranking model selects top-k relevant triplets from a knowledge graph
2. **Generation**: An LLM generates answers from the selected triplets

## Installation

```bash
pip install -r requirements.txt
```

Additionally, install [just](https://github.com/casey/just) command runner:
```bash
# macOS
brew install just

# Linux
cargo install just
```

## Configuration

All dataset paths and parameters are defined in `config.yml`. Update paths before running experiments.

---

## Commands

### Full Pipeline

Runs the complete workflow: Train → Triplet Selection → Coverage Analysis → LLM Inference.

```bash
just full-pipeline <dataset> [top-k] [sample-k] [llm]
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `dataset` | Dataset name (`cwq`, `webqsp`) | Required |
| `top-k` | Number of triplets to select | From config.yml |
| `sample-k` | Training sample size | 1000 |
| `llm` | LLM model (`llama`, `qwen`, `deepseek`) | From config.yml |

**Examples:**
```bash
just full-pipeline cwq                    # Use defaults from config.yml
just full-pipeline webqsp 50              # Override top-k
just full-pipeline cwq 100 1000 qwen      # Override top-k, sample-k, and LLM
```

**Output:** `results/full-pipeline/{dataset}/k{top-k}-N{sample-k}/`

---

### K-Ablation (KGScout)

Trains models for multiple k values and evaluates each.

```bash
just k-ablation <dataset>
```

K values are read from `config.yml`. For each k value, the pipeline:
1. Trains a model with that k
2. Generates triplets
3. Computes coverage metrics
4. Runs LLM inference

**Output:** `results/k-ablation/{dataset}/k{k}/`

---

### K-Ablation (Cosine Baseline)

Evaluates cosine similarity retriever (no training) across multiple k values.

```bash
just k-ablation-cosine <dataset> [llm]
```

**Examples:**
```bash
just k-ablation-cosine cwq
just k-ablation-cosine webqsp qwen
```

**Output:** `results/cosine-k-ablation/{dataset}/k{k}/`

---

### Model & Reward Ablation

Runs ablation studies for model architecture variants and reward function variants.

```bash
just run-ablations <dataset>
```

**Model variants:** `no-gate`, `no-ppr`, `no-ra`, `no-ta`  
**Reward variants:** `only_connection`, `only_presence`

**Output:** `results/ablation-2/{dataset}-model-ablation/` and `results/ablation-2/{dataset}-reward-ablation/`

---

### Statistical Analysis

Compares KGScout vs Cosine retrievers with case categorization.

```bash
just statistical-analysis <dataset> [k-values]
```

**Important:** Statistical analysis requires model checkpoints from prior training. Run `full-pipeline` or `k-ablation` first.

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `dataset` | Dataset name (`cwq`, `webqsp`) | Required |
| `k-values` | Space-separated k values (quoted) | `"30 50 100 150"` |

**Example:**
```bash
just statistical-analysis cwq
just statistical-analysis webqsp "30 50 100"
```

**Output:** `results/statistical-analysis/{dataset}/`

---

### Hop Analysis

Stratified analysis by reasoning hop count (1-hop, 2-hop, ≥3-hop).

```bash
just hop-analysis <dataset> [k-values]
```

**Important:** Requires model checkpoints from prior training. Run `full-pipeline` or `k-ablation` first.

**Example:**
```bash
just hop-analysis cwq
just hop-analysis webqsp "30 50 100 150"
```

**Output:** `results/hop-analysis/{dataset}/`

---

### Cross-Domain Transfer

Tests generalization by applying a model trained on one dataset to another.

```bash
just cross-domain <source-dataset> <target-dataset>
```

**Important:** Requires a trained model from `full-pipeline` on the source dataset.

**Example:**
```bash
just cross-domain cwq webqsp    # Train on CWQ, test on WebQSP
just cross-domain webqsp cwq    # Train on WebQSP, test on CWQ
```

**Output:** `results/crossdomain/src-{source}-target-{target}/`

---

## Results Directory Structure

```
results/
├── full-pipeline/
│   └── {dataset}/
│       └── k{top-k}-N{sample-k}/
│           ├── model/                    # Trained checkpoint
│           ├── triplet-analysis/
│           │   └── selected_triplets.json
│           ├── triplet_metrics/
│           │   └── coverage_metrics.json
│           └── {llm}-inference/
│               ├── llm_metrics.json
│               └── llm_detailed_results.json
│
├── k-ablation/
│   └── {dataset}/
│       └── k{k}/
│           ├── model/
│           ├── triplet-analysis/
│           ├── triplet_metrics/
│           └── model-result/
│
├── cosine-k-ablation/
│   └── {dataset}/
│       └── k{k}/
│           ├── triplet-analysis/
│           ├── triplet_metrics/
│           └── {llm}-inference/
│
├── ablation-2/
│   ├── {dataset}-model-ablation/
│   │   └── {variant}/                   # no-gate, no-ppr, no-ra, no-ta
│   │       ├── model/
│   │       ├── triplet-result/
│   │       ├── triplet_metrics/
│   │       └── llama-inference/
│   └── {dataset}-reward-ablation/
│       └── {variant}/                   # only_connection, only_presence
│           └── ...
│
├── statistical-analysis/
│   └── {dataset}/
│       ├── k{k}_statistical_analysis.json
│       └── summary.json
│
├── hop-analysis/
│   └── {dataset}/
│       ├── hop_labels.json
│       ├── k{k}_hop_analysis.json
│       └── summary.json
│
└── crossdomain/
    └── src-{source}-target-{target}/
        ├── triplet-result/
        ├── triplet_metrics/
        └── llama-inference/
```

---

## Output File Formats

### selected_triplets.json

Contains selected triplets for each question with metadata.

```json
[
  {
    "question": "...",
    "q_entity": ["..."],
    "a_entity": ["..."],
    "answer": ["..."],
    "selected_triplets": [["subject", "relation", "object"], ...]
  }
]
```

### coverage_metrics.json

Retrieval quality metrics.

```json
{
  "total_samples": 3531,
  "answer_coverage": 0.634,
  "path_coverage": 0.614,
  "answer_coverage_count": 2239,
  "path_coverage_count": 2167
}
```

| Metric | Description |
|--------|-------------|
| `answer_coverage` | Fraction of questions where answer entities appear in selected triplets |
| `path_connectivity` | Fraction of questions where a complete reasoning path exists |

### llm_metrics.json

LLM answer generation metrics.

```json
{
  "hit": 52.96,
  "hit_at_1": 43.44,
  "macro_f1": 39.23,
  "macro_precision": 40.48,
  "macro_recall": 46.90,
  "exact_match": 27.98,
  "totally_wrong": 47.04,
  "total_samples": 3531,
  "inference_time_seconds": 4463.31,
  "throughput_samples_per_sec": 0.79
}
```

| Metric | Description |
|--------|-------------|
| `hit` | % of questions with at least one correct answer |
| `macro_f1` | Macro-averaged F1 score |
| `exact_match` | % of questions with exact answer match |

### statistical_analysis.json

Case-wise comparison between Cosine and KGScout retrievers.

| Case | Description |
|------|-------------|
| Case 1 | Cosine finds no relevant triplets, KGScout finds some |
| Case 2 | Cosine finds relevant triplets but no path, KGScout finds path |
| Case 3 | Both find paths with high overlap (Jaccard ≥ 0.7) |
| Case 4 | Both find paths with low overlap (Jaccard ≤ 0.3) |
| Case 5 | Cosine outperforms KGScout |
| Case 6 | Both fail to find relevant triplets |

---

## Constraints & Dependencies

1. **Statistical analysis** and **hop analysis** require pre-trained model checkpoints. Run `full-pipeline` or `k-ablation` first.

2. **Cross-domain transfer** requires a trained model from `full-pipeline` on the source dataset.

3. All commands skip steps if output files already exist. Delete existing outputs to re-run.

4. Logs are saved to `logs/` directory.
