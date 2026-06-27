"""
Reward Function Ablation for Reversed Attention (ablation-2).

Uses ReversedOriginal model for all experiments. Varies the reward function:
  no_pres:   Remove w_pres * frac_presence
  no_conn:   Remove w_conn * conn_score
  no_path:   Remove w_cov * path_cov
  only_pres: Only frac_presence
  only_conn: Only conn_score
  only_cov:  Only path_cov

Results: ./results/ablation-2/reward-ablation/<variant>/
"""

import os
import sys
import json
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import logging
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.joint_dataset import JointTrainingDatasetv3PPR
import __main__
__main__.JointTrainingDatasetv3PPR = JointTrainingDatasetv3PPR

torch.manual_seed(100)
np.random.seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("ablation2.reward")

# Import the base reversed model and training infrastructure from model_ablation
# Use importlib since directory name has a hyphen
import importlib
_model_ablation = importlib.import_module("ablation-2.model_ablation")
ReversedOriginal = _model_ablation.ReversedOriginal
_sample_paths_impl = _model_ablation._sample_paths_impl
SampledJointTrainingDataset = _model_ablation.SampledJointTrainingDataset
CosinePretrainingDataset = _model_ablation.CosinePretrainingDataset
collate_fn_pretrain = _model_ablation.collate_fn_pretrain
CosinePretrainer = _model_ablation.CosinePretrainer
JointTrainer = _model_ablation.JointTrainer
generate_selected_json = _model_ablation.generate_selected_json
_find_last_checkpoint = _model_ablation._find_last_checkpoint


# ============================================================================
# REWARD VARIANTS
# ============================================================================

def _compute_components(triplets, q_entities, a_entities, lambda_lin=0.2):
    if not triplets:
        return 0.0, 0.0, 0.0
    G = nx.DiGraph()
    for s, p, o in triplets:
        G.add_edge(s.lower(), o.lower(), relation=p.lower())
    present = sum(1 for a in a_entities if a.lower() in G)
    frac_presence = present / len(a_entities) if a_entities else 0.0
    conn_score = 0.0
    for q in q_entities:
        for a in a_entities:
            try:
                d = nx.shortest_path_length(G, q.lower(), a.lower())
                conn_score = max(conn_score, max(0.0, 1.0 - lambda_lin * (d - 1)))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    triplet_pairs = {(s.lower(), o.lower()) for s, _, o in triplets}
    cov_scores = []
    for q in q_entities:
        for a in a_entities:
            try:
                path = nx.shortest_path(G, q.lower(), a.lower())
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) < 2: continue
            matches = sum(1 for u, v in zip(path, path[1:]) if (u, v) in triplet_pairs)
            cov_scores.append(matches / (len(path) - 1))
    path_cov = max(cov_scores) if cov_scores else 0.0
    return frac_presence, conn_score, path_cov


def reward_no_pres(triplets, q_entities, a_entities):
    _, conn, cov = _compute_components(triplets, q_entities, a_entities)
    return min(4 * conn + 3 * cov, 10.0)

def reward_no_conn(triplets, q_entities, a_entities):
    pres, _, cov = _compute_components(triplets, q_entities, a_entities)
    return min(3 * pres + 3 * cov, 10.0)

def reward_no_path(triplets, q_entities, a_entities):
    pres, conn, _ = _compute_components(triplets, q_entities, a_entities)
    return min(3 * pres + 4 * conn, 10.0)

def reward_only_pres(triplets, q_entities, a_entities):
    pres, _, _ = _compute_components(triplets, q_entities, a_entities)
    return pres

def reward_only_conn(triplets, q_entities, a_entities):
    _, conn, _ = _compute_components(triplets, q_entities, a_entities)
    return conn

def reward_only_cov(triplets, q_entities, a_entities):
    _, _, cov = _compute_components(triplets, q_entities, a_entities)
    return cov


# ============================================================================
# CONFIG & MAIN
# ============================================================================

REWARD_ABLATION_CONFIGS = {
    "no_pres":   {"reward_func": reward_no_pres,   "description": "Without frac_presence"},
    "no_conn":   {"reward_func": reward_no_conn,   "description": "Without conn_score"},
    "no_path":   {"reward_func": reward_no_path,   "description": "Without path_cov"},
    "only_pres": {"reward_func": reward_only_pres, "description": "Only frac_presence"},
    "only_conn": {"reward_func": reward_only_conn, "description": "Only conn_score"},
    "only_cov":  {"reward_func": reward_only_cov,  "description": "Only path_cov"},
}


def run_reward_ablation(train_path, val_path, test_path, output_base="./results/ablation-2/reward-ablation", experiments=None):
    logger.info("=" * 70)
    logger.info("ABLATION-2 REWARD FUNCTION STUDIES (Reversed Attention)")
    logger.info("=" * 70)
    configs = experiments if experiments else list(REWARD_ABLATION_CONFIGS.keys())

    # Phase 1: Train
    logger.info("PHASE 1: Training")
    train_data = torch.load(train_path, weights_only=False, map_location="cpu")
    val_data = torch.load(val_path, weights_only=False, map_location="cpu")
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    for name in configs:
        cfg = REWARD_ABLATION_CONFIGS[name]
        logger.info(f"{'='*60}\n  {name}: {cfg['description']}\n{'='*60}")
        exp_dir = os.path.join(output_base, name)
        pt_dir = os.path.join(exp_dir, "model", "pretrained")
        tr_dir = os.path.join(exp_dir, "model", "trained")
        os.makedirs(exp_dir, exist_ok=True)

        # Fresh model each time
        model = ReversedOriginal(device=str(device))
        # Pretrain
        logger.info(f"  [1/2] Pretrain n=500, 5 ep")
        pt_ds = CosinePretrainingDataset(train_data, k=500)
        pv_ds = CosinePretrainingDataset(val_data, k=500)
        pt_dl = DataLoader(pt_ds, batch_size=1, shuffle=True, collate_fn=collate_fn_pretrain)
        pv_dl = DataLoader(pv_ds, batch_size=1, shuffle=False, collate_fn=collate_fn_pretrain)
        CosinePretrainer(model, str(device), pt_dir).train(pt_dl, pv_dl, num_epochs=5)
        ckpt = torch.load(os.path.join(pt_dir, "best_pretrained_model-5.pt"), weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

        # Train with ablated reward
        logger.info(f"  [2/2] Train with {name} reward (k=1000, sample=100, 30 ep)")
        tr_ds = SampledJointTrainingDataset(train_data, k=1000)
        vl_ds = SampledJointTrainingDataset(val_data, k=1000)
        tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True)
        vl_dl = DataLoader(vl_ds, batch_size=1, shuffle=False)
        JointTrainer(model, cfg["reward_func"], checkpoint_dir=tr_dir).train(tr_dl, vl_dl, k=100)
        logger.info(f"  Saved: {tr_dir}")
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    del train_data, val_data; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Phase 2: Inference
    logger.info(f"{'='*70}\nPHASE 2: Inference\n{'='*70}")
    test_data = torch.load(test_path, weights_only=False, map_location="cpu")
    logger.info(f"  Test: {len(test_data)}")

    for name in configs:
        cfg = REWARD_ABLATION_CONFIGS[name]
        tr_dir = os.path.join(output_base, name, "model", "trained")
        res_dir = os.path.join(output_base, name, "triplet-result")
        ckpt_dir = _find_last_checkpoint(tr_dir)
        if not ckpt_dir:
            logger.warning(f"  {name}: no checkpoint, skip"); continue
        ckpt_path = os.path.join(ckpt_dir, "path_ranker.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  {name}: no path_ranker.pt, skip"); continue
        logger.info(f"  {name}: loading {ckpt_dir}")
        model = ReversedOriginal(device=str(device))
        state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        for key, val in state.items():
            if key in ('temperature', 'baseline'): getattr(model, key).data = val
            elif hasattr(model, key): getattr(model, key).load_state_dict(val)
            else: logger.warning(f"  Unexpected key: '{key}'")
        model.to(device)
        trainer = JointTrainer(model, cfg["reward_func"], checkpoint_dir=tr_dir)
        tst_ds = SampledJointTrainingDataset(test_data, k=1000)
        tst_dl = DataLoader(tst_ds, batch_size=1, shuffle=False)
        generate_selected_json(tst_dl, res_dir, trainer, top_k=100)
        del model, trainer; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("REWARD ABLATION COMPLETE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", default="./results/ablation-2/reward-ablation")
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_reward_ablation(args.train_data, args.val_data, args.test_data, args.output_dir, args.experiments)
