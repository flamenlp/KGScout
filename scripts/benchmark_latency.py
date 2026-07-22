"""
KGScout Inference Latency Benchmark
====================================
Benchmarks three stages of the retrieval pipeline:
  1. Cosine embedding creation (encoding question text with SentenceTransformer)
  2. Cosine similarity scoring + top-K selection (question vs all triplet embeddings)
  3. KGScout model inference + top-K selection

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --num_samples 50
"""

import argparse
import os
import sys
import time
import statistics

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model.path_ranker import PathRankingModel
from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR

# Allow loading datasets saved from notebooks
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# =============================================================================
# HARDCODED PATHS — update these for your environment
# =============================================================================
MODEL_PATH = "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/model/architecture-v8/v7-rv8-n1000-e30-k100_cosine/checkpoint_epoch_30"
DATA_PATH = "/mnt/hdd1/sourav23099/webqsp-v21/val/val_jointrainer_path_dataset_v3_ppr.pt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 100
SAMPLE_K = 1000
# =============================================================================


def load_model(model_path: str, device: str) -> PathRankingModel:
    """Load PathRankingModel from checkpoint directory."""
    print(f"Loading KGScout model from: {model_path}")
    model = PathRankingModel.from_pretrained(model_path, device=device)
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return model


def load_embedding_model():
    """Load the SentenceTransformer model used for cosine baseline."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    sbert = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")
    return sbert


def load_val_data(data_path: str, sample_k: int = 1000):
    """Load WebQSP validation dataset."""
    print(f"Loading validation data from: {data_path}")

    data = torch.load(data_path, weights_only=False, map_location="cpu")
    print(f"Loaded {len(data)} samples")

    from torch.utils.data import Dataset
    if isinstance(data, Dataset):
        base_dataset = data
    else:
        print("Wrapping raw data in JointTrainingDatasetv3PPR...")
        base_dataset = JointTrainingDatasetv3PPR(data, device="cpu")

    sampled_dataset = SampledJointTrainingDataset(base_dataset, k=sample_k)
    print(f"Dataset ready: {len(sampled_dataset)} samples, sample_k={sample_k}")
    return sampled_dataset


def compute_stats(latencies: list) -> dict:
    """Compute latency statistics from a list of per-sample latencies (in ms)."""
    n = len(latencies)
    if n == 0:
        return {}
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies) if n > 1 else 0.0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * n)] if n > 1 else latencies[0],
        "p99_ms": sorted(latencies)[int(0.99 * n)] if n > 1 else latencies[0],
        "total_s": sum(latencies) / 1000,
        "throughput_per_sec": n / (sum(latencies) / 1000) if sum(latencies) > 0 else 0,
    }


def benchmark(
    model: PathRankingModel,
    sbert,
    dataset,
    device: str,
    top_k: int = 100,
    num_samples: int = None,
    warmup: int = 5,
):
    """
    Run the full 3-stage benchmark.

    Returns dict with latency stats for each stage.
    """
    total_samples = len(dataset)
    if num_samples is not None:
        total_samples = min(num_samples, total_samples)

    print(f"\n{'='*60}")
    print(f"  LATENCY BENCHMARK — 3 Stages")
    print(f"{'='*60}")
    print(f"  Device:            {device}")
    print(f"  Samples:           {total_samples}")
    print(f"  Top-K:             {top_k}")
    print(f"  Embedding model:   {EMBEDDING_MODEL}")
    print(f"  Warmup iters:      {warmup}")
    print(f"{'='*60}\n")

    # ─── Warmup ──────────────────────────────────────────────────────────────
    print("Running warmup...")
    for i in range(min(warmup, total_samples)):
        sample = dataset[i]
        question_text = sample["question"]
        # Warmup SentenceTransformer
        _ = sbert.encode(question_text, convert_to_tensor=True)
        # Warmup KGScout model
        q_emb = sample["question_embedding"].to(device)
        t_emb = sample["topk_linearized_triplet_embeddings"].to(device)
        r_emb = sample["topK_rel_embeddings"].to(device)
        g_feat = sample["graph_features"].to(device)
        with torch.no_grad():
            _ = model(q_emb, t_emb, r_emb, g_feat)

    if device == "cuda":
        torch.cuda.synchronize()

    # ─── Timed Benchmark ─────────────────────────────────────────────────────
    cosine_embed_latencies = []     # Stage 1: question embedding creation
    cosine_score_latencies = []     # Stage 2: cosine similarity + topK
    kgscout_latencies = []          # Stage 3: KGScout model forward + topK
    num_triplets_per_sample = []

    for i in tqdm(range(total_samples), desc="Benchmarking"):
        sample = dataset[i]
        question_text = sample["question"]
        triplet_embeds = sample["topk_linearized_triplet_embeddings"].to(device)
        relation_embeds = sample["topK_rel_embeddings"].to(device)
        graph_features = sample["graph_features"].to(device)

        num_triplets = triplet_embeds.shape[0]
        num_triplets_per_sample.append(num_triplets)

        if device == "cuda":
            torch.cuda.synchronize()

        # ── Stage 1: Cosine Embedding Creation ───────────────────────────────
        start = time.perf_counter()

        question_embedding = sbert.encode(question_text, convert_to_tensor=True)
        question_embedding = question_embedding.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        cosine_embed_latencies.append((end - start) * 1000)

        # ── Stage 2: Cosine Similarity Scoring + Top-K ───────────────────────
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Normalize and compute cosine similarity
        q_norm = F.normalize(question_embedding.unsqueeze(0), p=2, dim=-1)  # (1, 384)
        t_norm = F.normalize(triplet_embeds, p=2, dim=-1)                    # (N, 384)
        cosine_scores = torch.mm(q_norm, t_norm.t()).squeeze(0)              # (N,)

        # Top-K selection by cosine score
        k = min(top_k, len(cosine_scores))
        _ = torch.topk(cosine_scores, k)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        cosine_score_latencies.append((end - start) * 1000)

        # ── Stage 3: KGScout Model Inference + Top-K ─────────────────────────
        # Use the precomputed question embedding from dataset (matches training)
        q_embed_model = sample["question_embedding"].to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            ranking_scores, path_probs = model(
                q_embed_model, triplet_embeds, relation_embeds, graph_features
            )
            k = min(top_k, len(ranking_scores))
            _ = torch.topk(ranking_scores, k)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        kgscout_latencies.append((end - start) * 1000)

    # ─── Compute Statistics ──────────────────────────────────────────────────
    results = {
        "num_samples": total_samples,
        "device": device,
        "top_k": top_k,
        "avg_triplets_per_sample": statistics.mean(num_triplets_per_sample),
        "stage1_cosine_embedding": compute_stats(cosine_embed_latencies),
        "stage2_cosine_scoring": compute_stats(cosine_score_latencies),
        "stage3_kgscout_inference": compute_stats(kgscout_latencies),
    }

    return results


def print_stage_stats(title: str, stats: dict):
    """Print stats for one stage."""
    print(f"  {title}")
    print(f"  {'─'*54}")
    print(f"    Mean:        {stats['mean_ms']:.3f} ms")
    print(f"    Median:      {stats['median_ms']:.3f} ms")
    print(f"    Std:         {stats['std_ms']:.3f} ms")
    print(f"    Min:         {stats['min_ms']:.3f} ms")
    print(f"    Max:         {stats['max_ms']:.3f} ms")
    print(f"    P95:         {stats['p95_ms']:.3f} ms")
    print(f"    P99:         {stats['p99_ms']:.3f} ms")
    print(f"    Total:       {stats['total_s']:.2f} s")
    print(f"    Throughput:  {stats['throughput_per_sec']:.1f} samples/sec")
    print()


def print_results(results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  LATENCY BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Device:                {results['device']}")
    print(f"  Samples evaluated:     {results['num_samples']}")
    print(f"  Top-K:                 {results['top_k']}")
    print(f"  Avg triplets/sample:   {results['avg_triplets_per_sample']:.1f}")
    print(f"{'='*60}\n")

    print_stage_stats(
        "Stage 1: Cosine Embedding Creation (SentenceTransformer encode)",
        results["stage1_cosine_embedding"],
    )
    print_stage_stats(
        "Stage 2: Cosine Similarity Scoring + Top-K Selection",
        results["stage2_cosine_scoring"],
    )
    print_stage_stats(
        "Stage 3: KGScout Model Inference + Top-K Selection",
        results["stage3_kgscout_inference"],
    )

    # Summary comparison
    s1 = results["stage1_cosine_embedding"]["mean_ms"]
    s2 = results["stage2_cosine_scoring"]["mean_ms"]
    s3 = results["stage3_kgscout_inference"]["mean_ms"]
    cosine_total = s1 + s2

    print(f"{'='*60}")
    print(f"  SUMMARY (Mean Latency)")
    print(f"{'─'*60}")
    print(f"    Cosine Embedding:          {s1:.3f} ms")
    print(f"    Cosine Scoring + Top-K:    {s2:.3f} ms")
    print(f"    Cosine Total (S1 + S2):    {cosine_total:.3f} ms")
    print(f"    KGScout Inference + Top-K: {s3:.3f} ms")
    print(f"{'─'*60}")
    print(f"    KGScout overhead vs Cosine Scoring: {s3 - s2:.3f} ms ({(s3/s2):.2f}x)" if s2 > 0 else "")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="KGScout Inference Latency Benchmark")
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Limit to first N samples (default: all)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on ('cuda' or 'cpu'). Auto-detects if not set."
    )
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load models
    kgscout_model = load_model(MODEL_PATH, device)
    sbert = load_embedding_model()

    # Load data
    dataset = load_val_data(DATA_PATH, sample_k=SAMPLE_K)

    # Run benchmark
    results = benchmark(
        model=kgscout_model,
        sbert=sbert,
        dataset=dataset,
        device=device,
        top_k=TOP_K,
        num_samples=args.num_samples,
        warmup=args.warmup,
    )

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
