#!/usr/bin/env python3
"""
Compute average, min, and max number of tokens for the full LLM prompt
built from a selected_triplets.json file.

Usage:
    python compute_prompt_tokens.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_inference import format_prompt

# ============================================================================
# CONFIGURATION — edit these values directly
# ============================================================================
INPUT_PATH = "no_pres/triplet-result/selected_triplets.json"
TOP_K = 100
TOKENIZER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# ============================================================================


def load_tokenizer(model_id: str):
    """Load a HuggingFace tokenizer. Falls back to simple whitespace split if unavailable."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Loaded tokenizer: {model_id}")
        return tokenizer.encode
    except Exception as e:
        print(f"WARNING: Could not load tokenizer '{model_id}': {e}")
        print("Falling back to whitespace-based token estimation.")
        return None


def whitespace_token_count(text: str) -> int:
    """Rough token estimate using whitespace split (typically underestimates by ~25%)."""
    return len(text.split())


def main():
    # Load data
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {INPUT_PATH}")
    print(f"Using top_k = {TOP_K}")

    # Load tokenizer
    encode_fn = load_tokenizer(TOKENIZER_MODEL)
    if encode_fn is None:
        count_fn = whitespace_token_count
        method = "whitespace-split (approximate)"
    else:
        count_fn = lambda text: len(encode_fn(text))
        method = f"tokenizer ({TOKENIZER_MODEL})"

    print(f"Token counting method: {method}\n")

    # Compute token counts for each prompt
    token_counts = []
    for i, sample in enumerate(data):
        question = sample["question"]
        triplets = sample.get("reranker", [])
        q_entity = sample.get("q_entity", [])

        prompt = format_prompt(question, triplets, topk=TOP_K, q_entity=q_entity)
        n_tokens = count_fn(prompt)
        token_counts.append(n_tokens)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples...")

    # Compute statistics
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    print("=" * 50)
    print("PROMPT TOKEN STATISTICS")
    print("=" * 50)
    print(f"  Samples:  {len(token_counts)}")
    print(f"  Top-K:    {TOP_K}")
    print(f"  Method:   {method}")
    print("-" * 50)
    print(f"  Average:  {avg_tokens:.1f} tokens")
    print(f"  Min:      {min_tokens} tokens")
    print(f"  Max:      {max_tokens} tokens")
    print("=" * 50)


if __name__ == "__main__":
    main()
