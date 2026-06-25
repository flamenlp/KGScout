# Ablation-2: Reversed Attention Architecture Study
# Tests the effect of swapping Query/Key-Value in the cross-attention modules.
# Original: Query=Triplets, Key=Value=Question (output identical across rows)
# Reversed: Query=Question, Key=Value=Triplets (output is question attending over triplets)
