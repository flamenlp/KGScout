#!/usr/bin/env python3
"""
LLaMA Inference on Ablation Results.

Loads Meta-Llama-3.1-8B-Instruct once, iterates over all ablation variants'
selected_triplets.json files, generates answers using top-100 triplets with ICL prompt,
and computes QA metrics (Hit, Hit@1, F1, Precision, Recall, Exact Match).

Results saved to: ./results/<ablation-type>/<variant>/llama-inference/

Usage:
    python run_ablation_inference.py                          # all variants
    python run_ablation_inference.py --mode model             # model ablations only
    python run_ablation_inference.py --mode reward            # reward ablations only
    python run_ablation_inference.py --mode model --experiments no-ppr no-gate
"""

import os
import sys
import json
import re
import string
import argparse
import logging
import time
import numpy as np
from copy import deepcopy
from typing import List
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# HARDCODED CONFIGURATION
# ============================================================================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_ABLATION_DIR = "./results/model-ablation"
REWARD_ABLATION_DIR = "./results/reward-ablation"
TOP_K = 100
LOG_FILE = os.path.join("logs", "ablation_inference.log")

MODEL_VARIANTS = ["no-ppr", "no-rt", "no-tt", "no-gate", "no-ra", "no-ta"]
REWARD_VARIANTS = ["no_pres", "no_conn", "no_path", "only_pres", "only_conn", "only_cov"]

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("ablation.inference")


def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


# ============================================================================
# ICL PROMPT TEMPLATES
# ============================================================================

icl_user_prompt1 = """
Triplets:
- Lou Seal, sports.mascot.team, San Francisco Giants
- San Francisco Giants, sports.sports_team.championships, 2012 World Series
- San Francisco Giants, sports.sports_team.championships, 2010 World Series
- San Francisco Giants, sports.sports_team.championships, 2014 World Series
- Crazy Crab, sports.mascot.team, San Francisco Giants
- San Francisco Giants, sports.professional_sports_team.owner_s, Bill Gates
- New York Yankees, sports.sports_team.championships, 2009 World Series
- San Francisco Giants, sports.sports_team.colors, Blue and White
- m.0k079qm, base.schemastaging.team_training_ground_relationship.team, San Francisco Giants
- m.0k079ry, base.schemastaging.team_training_ground_relationship.team, San Francisco Giants

Question:
What year did the team with mascot named Lou Seal win the World Series?"""

icl_ass_prompt1 = """Answer in JSON format:
{"ans" : ["2014 (2014 World Series)", "2012 (2012 World Series)", "2010 (2010 World Series)"] }

Reason:
To answer the question, we need to:
1. Identify the team with the mascot Lou Seal.
2. Find the years that team won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.
Now, we need to find the year the San Francisco Giants won the World Series.
From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.
So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the team with mascot Lou Seal won the World Series in 2010, 2012, and 2014.
"""

icl_user_prompt2 = """
Triplets: 
- Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
- Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
- Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group

Question:
Who is the coach of the team owned by Steve Bisciotti?"""

icl_ass_prompt2 = """Answer in JSON format:
{"ans": ["answer not available"]}

Reason:
Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.
"""


def create_prompt_icl(question, triplets, topk):
    triplet_text = "\n".join([f"- {triplet}" for triplet in triplets[:topk]])
    na_format = '{"ans": ["answer not available"]}'
    ans_format = '{"ans":["your answer 1","your answer 2"]}'
    user_query = f"""
Linearized Triplets: 
{triplet_text}

Question:
{question}

Let's think step by step."""

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a knowledge graph question answering system. Given a question and relevant linearized knowledge triplets, provide correct answers in JSON format supported by the triplets and provide a brief reason for the answer based on the triplets.

Instructions:
1> Provide your final answer in JSON format using this placeholder: {ans_format}. If there is insufficient information to answer the question, return {na_format}.
2> Ensure answer does not contain duplicate entries.
3> Keep your reasoning brief and focused.
4> Your answer must not contradict any information presented in the provided triplets.
5> If the answer is directly supported by the triplets, use the triplets to justify your answer.
6> If the answer is not explicitly found in the triplets, you may use your own factual knowledge but only if it is consistent with the information in the triplets.

#Example 1:
{icl_user_prompt1}
#Answer:
{icl_ass_prompt1}

#Example 2:
{icl_user_prompt2}
#Answer:
{icl_ass_prompt2}

Now consider the below Triplets and answer the Question carefully
<|start_header_id|>user<|end_header_id|>\n
{user_query}
#Answer:
<|start_header_id|>assistant<|end_header_id|>
    """
    return prompt


# ============================================================================
# EVALUATION FUNCTIONS (from notebooks/lama-inference.py)
# ============================================================================

def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_pred(prediction):
    pattern = r'\{[^{}]*\}'
    json_match = re.search(pattern, prediction)
    if not json_match:
        if '{"ans": [' in prediction:
            x = prediction.split('{"ans": [')[-1]
            entities = [ele.strip().strip('"').lower() for ele in x.split(",")]
            response = remove_duplicates(entities)
        else:
            response = [prediction.strip()]
    else:
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
            response = remove_duplicates(data["ans"])
        except Exception:
            response = [prediction.strip()]
    return response


def eval_recall(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / len(answer), matched, len(answer)


def eval_precision(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    num_pred = len(prediction)
    if num_pred == 0:
        return 0, 0, 0
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / num_pred, matched, num_pred


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def eval_hit1(prediction, answer, double_check):
    if len(prediction) == 0:
        return 0
    for a in answer:
        if match(prediction[0], a):
            return 1
        elif double_check:
            if match(a, prediction[0].strip()):
                return 1
    return 0


def eval_hit(prediction, answer, double_check):
    if len(prediction) == 0:
        return 0
    for a in answer:
        for p in prediction:
            if match(p, a):
                return 1
            elif double_check and match(a, p.strip()):
                return 1
    return 0


def generate_answer(model, tokenizer, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def evaluate_dataset(data, output_dir, model, tokenizer, top_k):
    """Run LLaMA inference on selected triplets and compute QA metrics."""
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, 'predictions.jsonl')
    detailed_results_file = os.path.join(output_dir, 'detailed_results.jsonl')

    hit_list, hit1_list, f1_list = [], [], []
    precision_list, recall_list = [], []
    total_pred, total_answer, total_match = 0, 0, 0

    with open(predictions_file, 'w', encoding='utf-8') as pred_f, \
         open(detailed_results_file, 'w', encoding='utf-8') as detail_f:

        for i, datapoint in enumerate(tqdm(data, desc="    LLaMA Inference")):
            try:
                question = datapoint["question"]
                a_entity = datapoint["a_entity"]
                sorted_paths = datapoint["reranker"]

                if not a_entity or len(a_entity) == 0:
                    continue

                prompt = create_prompt_icl(question, sorted_paths, top_k)
                raw_prediction = generate_answer(model, tokenizer, prompt)
                prediction_r = get_pred(raw_prediction)
                prediction = [s for s in prediction_r if s != ""]

                answer = sorted(remove_duplicates(a_entity), key=len, reverse=True)
                if 'when' in question.lower() or 'what year' in question.lower():
                    for idx in range(len(answer)):
                        if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                            answer[idx] = answer[idx].split('-')[0]

                double_check = any(kw in question.lower() for kw in
                                   ['when', 'what year', 'which year', 'where', 'sport',
                                    "what countr", "language", 'nba finals', 'world series'])

                precision_score, matched_1, num_pred = eval_precision(prediction, answer, double_check)
                recall_score, matched_2, num_answer = eval_recall(prediction, answer, double_check)
                f1_score = eval_f1(precision_score, recall_score)
                hit1 = eval_hit1(prediction, answer, double_check)
                hit = eval_hit(prediction, answer, double_check)

                assert matched_1 == matched_2
                total_pred += num_pred
                total_answer += num_answer
                total_match += matched_1

                hit1_list.append(hit1)
                hit_list.append(hit)
                f1_list.append(f1_score)
                precision_list.append(precision_score)
                recall_list.append(recall_score)

                pred_data = {
                    'id': i, 'question': question,
                    'prediction': raw_prediction,
                    'processed_prediction': prediction,
                    'ground_truth': answer
                }
                pred_f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')

                detail_data = {
                    'id': i, 'question': question,
                    'prediction': prediction, 'ground_truth': answer,
                    'hit@1': hit1, 'hit': hit,
                    'f1': f1_score, 'precision': precision_score, 'recall': recall_score
                }
                detail_f.write(json.dumps(detail_data, ensure_ascii=False) + '\n')

            except Exception as e:
                logger.warning(f"  Error processing item {i}: {e}")
                continue

    if len(hit_list) == 0:
        logger.warning("  No valid predictions found!")
        return None

    avg_hit = sum(hit_list) * 100 / len(hit_list)
    avg_hit1 = sum(hit1_list) * 100 / len(hit1_list)
    avg_f1 = sum(f1_list) * 100 / len(f1_list)
    avg_precision = sum(precision_list) * 100 / len(precision_list)
    avg_recall = sum(recall_list) * 100 / len(recall_list)
    num_exact_match = (np.array(f1_list) == 1).sum() / len(f1_list) * 100
    num_totally_wrong = (np.array(recall_list) == 0).sum() / len(recall_list) * 100
    micro_precision = total_match / total_pred if total_pred > 0 else 0
    micro_recall = total_match / total_answer if total_answer > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    result_str = (f"Hit: {avg_hit:.2f}, Hit@1: {avg_hit1:.2f}, Macro F1: {avg_f1:.2f}, "
                  f"Macro Precision: {avg_precision:.2f}, Macro Recall: {avg_recall:.2f}, "
                  f"Exact Match: {num_exact_match:.2f}, Totally Wrong: {num_totally_wrong:.2f}")
    logger.info(f"    {result_str}")
    logger.info(f"    Micro F1: {micro_f1:.4f}, Micro P: {micro_precision:.4f}, Micro R: {micro_recall:.4f}")
    logger.info(f"    Total samples: {len(hit_list)}")

    results_file = os.path.join(output_dir, 'final_results.txt')
    with open(results_file, 'w') as f:
        f.write(result_str + '\n')
        f.write(f"Micro F1: {micro_f1:.4f}, Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}\n")
        f.write(f"Total samples: {len(hit_list)}\n")

    return {
        'hit': avg_hit, 'hit@1': avg_hit1,
        'macro_f1': avg_f1, 'macro_precision': avg_precision, 'macro_recall': avg_recall,
        'exact_match': num_exact_match, 'totally_wrong': num_totally_wrong,
        'micro_f1': micro_f1, 'micro_precision': micro_precision, 'micro_recall': micro_recall
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run LLaMA inference on ablation results")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "model", "reward"])
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("ABLATION INFERENCE: LLaMA QA Evaluation")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Top-K: {TOP_K}")

    # Load LLaMA model once
    logger.info("Loading LLaMA model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for param in model.parameters():
        param.requires_grad = False
    logger.info("  LLaMA model loaded.")

    # Build list of (ablation_type_dir, variant_name) to process
    tasks = []
    if args.mode in ("all", "model"):
        variants = args.experiments if (args.experiments and args.mode == "model") else MODEL_VARIANTS
        for v in variants:
            tasks.append((MODEL_ABLATION_DIR, v))
    if args.mode in ("all", "reward"):
        variants = args.experiments if (args.experiments and args.mode == "reward") else REWARD_VARIANTS
        for v in variants:
            tasks.append((REWARD_ABLATION_DIR, v))

    for base_dir, variant in tasks:
        input_path = os.path.join(base_dir, variant, "triplet-result", "selected_triplets.json")
        output_path = os.path.join(base_dir, variant, "llama-inference")

        logger.info(f"{'='*60}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Input:  {input_path}")
        logger.info(f"  Output: {output_path}")

        if not os.path.exists(input_path):
            logger.warning(f"  SKIPPED: {input_path} not found")
            continue

        with open(input_path, "r") as f:
            data = json.load(f)
        logger.info(f"  Loaded {len(data)} samples")

        evaluate_dataset(data, output_path, model, tokenizer, TOP_K)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"ALL INFERENCE COMPLETE. Total time: {elapsed / 3600:.2f} hours")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
