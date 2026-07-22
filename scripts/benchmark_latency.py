"""
KGScout Inference Latency Benchmark
====================================
Loads a trained PathRankingModel checkpoint and WebQSP validation data,
then measures per-sample inference latency.
"""

import argparse
import os
import sys
import time
import statistics

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model.path_ranker import PathRankingModel
from src.preprocess.sampled_dataset import SampledJointTrainingDataset
from src.preprocess.joint_dataset import JointTrainingDatasetv3PPR

# Allow loading datasets saved from notebooks where JointTrainingDatasetv3PPR
# was defined in __main__. torch.load/unpickle looks up __main__ for the class.
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

# =============================================================================
# HARDCODED PATHS — update these for your environment
# =============================================================================
MODEL_PATH = "/mnt/LS226/LS25/sourav23099/webqsp/webqsp-v21/model/architecture-v8/v7-rv8-n1000-e30-k100_cosine/checkpoint_epoch_30"
DATA_PATH = "/mnt/hdd1/sourav23099/webqsp-v21/val/val_jointrainer_path_dataset_v3_ppr.pt"
TOP_K = 100
SAMPLE_K = 1000
# =============================================================================


def load_model(model_path: str, device: str) -> PathRankingModel:
    """Load PathRankingModel from checkpoint directory."""
    print(f"Loading model from: {model_path}")
    model = PathRankingModel.from_pretrained(model_path, device=device)
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return model


def load_val_data(data_path: str, sample_k: int = 1000):
    """
    Load WebQSP validation dataset.
    
    Returns a SampledJointTrainingDataset wrapping the raw .pt data.
    """
    print(f"Loading validation data from: {data_path}")

    # Register class for unpickling
    import __main__
    __main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

    data = torch.load(data_path, weights_only=False, map_location="cpu")
    print(f"Loaded {len(data)} samples")

    from torch.utils.data import Dataset
    if isinstance(data, Dataset):
        base_dataset = data
    else:
        print("Wrapping raw data in JointTrainingDatasetv3PPR...")
        base_dataset = JointTrainingDatasetv3PPR(data, device="cpu")

    # Wrap in SampledJointTrainingDataset to match inference pipeline
    sampled_dataset = SampledJointTrainingDataset(base_dataset, k=sample_k)
    print(f"Dataset ready: {len(sampled_dataset)} samples, sample_k={sample_k}")
    return sampled_dataset


def benchmark_inference(
    model: PathRankingModel,
    dataset,
    device: str,
    top_k: int = 100,
    num_samples: int = None,
    warmup: int = 5,
):
    """
    Benchmark inference latency on the dataset.

    Args:
        model: Loaded PathRankingModel in eval mode.
        dataset: SampledJointTrainingDataset.
        device: 'cuda' or 'cpu'.
        top_k: Number of top triplets to select per question.
        num_samples: Limit evaluation to first N samples (None = all).
        warmup: Number of warmup iterations before timing.

    Returns:
        Dict with latency statistics.
    """
    total_samples = len(dataset)
    if num_samples is not None:
        total_samples = min(num_samples, total_samples)

    print(f"\n{'='*60}")
    print(f"Benchmarking Inference Latency")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Samples:      {total_samples}")
    print(f"  Top-K:        {top_k}")
    print(f"  Warmup iters: {warmup}")
    print(f"{'='*60}\n")

    # Warmup (to stabilize GPU clocks, JIT compilation, etc.)
    print("Running warmup...")
    for i in range(min(warmup, total_samples)):
        sample = dataset[i]
        question_embed = sample["question_embedding"].to(device)
        triplet_embeds = sample["topk_linearized_triplet_embeddings"].to(device)
        relation_embeds = sample["topK_rel_embeddings"].to(device)
        graph_features = sample["graph_features"].to(device)

        with torch.no_grad():
            _ = model(question_embed, triplet_embeds, relation_embeds, graph_features)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed inference
    latencies = []
    num_triplets_per_sample = []

    for i in tqdm(range(total_samples), desc="Inference"):
        sample = dataset[i]

        question_embed = sample["question_embedding"].to(device)
        triplet_embeds = sample["topk_linearized_triplet_embeddings"].to(device)
        relation_embeds = sample["topK_rel_embeddings"].to(device)
        graph_features = sample["graph_features"].to(device)

        num_triplets = triplet_embeds.shape[0]
        num_triplets_per_sample.append(num_triplets)

        # Synchronize before timing (for accurate GPU measurement)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            ranking_scores, path_probs = model(
                question_embed, triplet_embeds, relation_embeds, graph_features
            )
            # Top-k selection (part of inference pipeline)
            k = min(top_k, len(ranking_scores))
            _ = torch.topk(ranking_scores, k)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    # Compute statistics
    results = {
        "num_samples": total_samples,
        "device": device,
        "top_k": top_k,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))],
        "p99_latency_ms": sorted(latencies)[int(0.99 * len(latencies))],
        "total_time_s": sum(latencies) / 1000,
        "throughput_samples_per_sec": total_samples / (sum(latencies) / 1000),
        "avg_triplets_per_sample": statistics.mean(num_triplets_per_sample),
    }

    return results


def print_results(results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  LATENCY BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Device:                  {results['device']}")
    print(f"  Samples evaluated:       {results['num_samples']}")
    print(f"  Top-K selection:         {results['top_k']}")
    print(f"  Avg triplets/sample:     {results['avg_triplets_per_sample']:.1f}")
    print(f"{'─'*60}")
    print(f"  Mean latency:            {results['mean_latency_ms']:.3f} ms")
    print(f"  Median latency:          {results['median_latency_ms']:.3f} ms")
    print(f"  Std deviation:           {results['std_latency_ms']:.3f} ms")
    print(f"  Min latency:             {results['min_latency_ms']:.3f} ms")
    print(f"  Max latency:             {results['max_latency_ms']:.3f} ms")
    print(f"  P95 latency:             {results['p95_latency_ms']:.3f} ms")
    print(f"  P99 latency:             {results['p99_latency_ms']:.3f} ms")
    print(f"{'─'*60}")
    print(f"  Total time:              {results['total_time_s']:.2f} s")
    print(f"  Throughput:              {results['throughput_samples_per_sec']:.1f} samples/sec")
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

    # Load model
    model = load_model(MODEL_PATH, device)

    # Load data
    dataset = load_val_data(DATA_PATH, sample_k=SAMPLE_K)

    # Run benchmark
    results = benchmark_inference(
        model=model,
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
