#!/usr/bin/env python3
"""Quick inspection of preprocessed MetaQA .pt file."""

import sys
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inspect preprocessed MetaQA data")
    parser.add_argument("--path", type=str, default="data/metaqa/processed/metaqa-1hop-test.pt",
                        help="Path to .pt file")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to print")
    args = parser.parse_args()

    print(f"Loading: {args.path}")
    data = torch.load(args.path, map_location="cpu", weights_only=False)
    print(f"Total samples: {len(data)}\n")

    for i in range(min(args.n, len(data))):
        sample = data[i]
        print(f"{'=' * 60}")
        print(f"SAMPLE {i}")
        print(f"{'=' * 60}")
        print(f"Question:    {sample['question']}")
        print(f"Q Entities:  {sample['q_entity']}")
        print(f"A Entities:  {sample['a_entity']}")
        print(f"Answers:     {sample['answer']}")
        print(f"Is Empty:    {sample['is_empty']}")
        print(f"\nQuestion Embedding shape: {sample['question_embedding'].shape}")
        print(f"Triplet Embeddings shape: {sample['topk_linearized_triplet_embeddings'].shape}")
        print(f"Relation Embeddings shape: {sample['topK_rel_embeddings'].shape}")
        print(f"Num triplets: {len(sample['topk_linearized_triplets'])}")
        print(f"Num rel_data: {len(sample['topk_rel_data'])}")

        print(f"\nFirst 5 linearized triplets:")
        for t in sample['topk_linearized_triplets'][:5]:
            print(f"  - {t}")

        print(f"\nFirst 5 topk_rel_data (score, triplet):")
        for score, triplet in sample['topk_rel_data'][:5]:
            print(f"  - score={score:.4f}  triplet={triplet}")

        print()


if __name__ == "__main__":
    main()
