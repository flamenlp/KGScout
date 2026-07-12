"""
Reward Function Ablation for KGScout.

The full reward has 2 components with weights:
  - Entity Presence: weight 4 (frac_presence)
  - Connectivity:    weight 6 (conn_score)
  - Total: min(4*presence + 6*connection, 10.0)

Ablation variants (defined in src/training/rewards.py):
  only_presence:   Only entity presence component → min(4*presence, 10.0)
  only_connection: Only connectivity component   → min(6*connection, 10.0)

Usage:
  These reward functions are used via the CLI:
    python cli.py train --reward-function only_presence ...
    python cli.py train --reward-function only_connection ...

  The model used for reward ablation is the default PathRankingModel from src/.
"""

# Re-export for reference (actual implementations live in src/training/rewards.py)
from src.training.rewards import (
    compute_reward_v8,
    reward_only_presence,
    reward_only_connection,
    get_reward_function,
    REWARD_FUNCTIONS,
)
