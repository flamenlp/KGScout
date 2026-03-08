import json
import re
import string
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




icl_user_prompt1 = """
Triplets:
- Lou Seal, sports.mascot.team, San Francisco Giants
- San Francisco Giants,  sports sports team championships, 2012 World Series
- San Francisco Giants, sports sports team championships 2010 World Series
- San Francisco Giants, sports sports team championships, 2014 World Series
- Crazy Crab, sports mascot team, San Francisco Giants
- San Francisco Giants, sports professional sports team owner s, Bill Gates
- New York Yankees, sports sports team championships, 2009 World Series
- San Francisco Giants, sports sports team colors, Blue and White
- m.0k079qm, base schemastaging team training ground relationship team, San Francisco Giants
- m.0k079ry, base.schemastaging team training ground relationship team, San Francisco Giants

Question:
What year did the team with mascot named Lou Seal win the World Series?"""

icl_ass_prompt1  = """Answer in JSON format:
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
- Steve Bisciotti, sports professional sports team owner s, Baltimore Ravens
- Steve Bisciotti, sports sports team owner teams owned, Baltimore Ravens
- Steve Bisciotti, organization organization founder organizations founded, Allegis Group

Question:
Who is the coach of the team owned by Steve Bisciotti?"""

icl_ass_prompt2="""Answer in JSON format:
{"ans": ["answer not available"]}

Reason:
Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.
"""



def create_prompt_icl(question, triplets, topk):
#     triplet_text = "\n".join([f"- ({triplet[0]}, {triplet[1]}, {triplet[2]})" for triplet in triplets[:topk]])
    triplet_text = "\n".join([f"- {triplet}" for triplet in triplets[:topk]])
    na_format = '{\"ans\": [\"answer not available\"]}'
    ans_format = '{\"ans\":[\"your answer 1\",\"your answer 2\"]}'
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
6> If the answer is not explicitly found in the triplets, you may use your own factual knowledgebut only if it is consistent with the information in the triplets.

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



"""
Script to evaluate Llama 3.1-8B on KGQA using linearized triplets sorted by cosine similarity.
Computes Macro-F1, Hit@1, Precision, Recall, and Exact Match metrics.
"""

json_match_error=[]
json_decode_error=[]
data_error=[]

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    """Check if s2 is contained in s1 after normalization."""
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def remove_duplicates(input_list):
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_pred(prediction):
    """Extract predictions from model output."""
    response_error=0
    decode_error=0
    global json_match_error, json_decode_error, data_error
    pattern = r'\{[^{}]*\}'
    json_match = re.search(pattern, prediction)
    if not json_match:
        print("###############################>>No JSON match")
        if '{"ans": [' in prediction:
            x = prediction.split("""{"ans": [""")[-1]
            entities = [ele.strip().strip('"').lower() for ele in x.split(",")]
            response = remove_duplicates(entities)
            #print("NJM response: ", response)
        else: 
            # lines = prediction.strip().split('\n')
            # if lines:
            #     if len(lines)>=2:
            #         response = [lines[-1], lines[-2]]# Take the last line as answer
            #     else:
            #         response = [lines[-1]]
            #     print("NJM response no ans found: ", response)
            #     json_match_error.append({"response":prediction})
            #     response_error+=1
            response = [prediction.strip()]
            print("NJM response no ans found: ", response)
    else:
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
            response = remove_duplicates(data["ans"])
            #print(response)
        except Exception as e:
            print("###############################>>Error decoding JSON")
            json_decode_error.append({"response":prediction})
            decode_error+=1
            # lines = prediction.strip().split('\n')
            # if len(lines)>=2:
            #     response = [lines[-1], lines[-2]]# Take the last line as answer
            # else:
            #     response = [lines[-1]]
            response = [prediction.strip()]
            print("JDE response: ", response)
    if response_error>0 or decode_error>0:
        data_error.append((response_error,decode_error))
    return response


def eval_recall(prediction, answer, double_check):
    """Calculate recall score."""
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
    """Calculate precision score."""
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
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def eval_hit1(prediction, answer, double_check):
    """Calculate Hit@1 score."""
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
    """Calculate Hit score."""
    if len(prediction) == 0:
        return 0
    for a in answer:
        for p in prediction:
            if match(p, a):
                return 1
            elif double_check and match(a,p.strip()):
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
    #print(inputs['input_ids'].shape[1])
    #pl=len(tokenizer.encode(prompt))
    #print(pl)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_dataset(data, output_dir, top_k): 
    os.makedirs(output_dir, exist_ok=True)

    global model, tokenizer
    predictions_file = os.path.join(output_dir, 'predictions.jsonl')
    detailed_results_file = os.path.join(output_dir, 'detailed_results.jsonl')
    hit_list = []
    hit1_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    total_pred = 0
    total_answer = 0
    total_match = 0
    with open(predictions_file, 'w', encoding='utf-8') as pred_f, \
         open(detailed_results_file, 'w', encoding='utf-8') as detail_f:
        
        for i, datapoint in enumerate(tqdm(data)):
            try:
                question = datapoint["question"]
                answer = datapoint["answer"]
                a_entity = datapoint["a_entity"]
                sorted_paths = datapoint["reranker"]
                if a_entity == [] or len(a_entity)==0:
                    continue
                prompt = create_prompt_icl(question, sorted_paths, top_k)
                #prompt = create_prompt_icl(question)
                raw_prediction = generate_answer(model, tokenizer, prompt)
                #print(prompt)
                prediction_r = get_pred(raw_prediction)
                prediction = [s for s in prediction_r if s != ""]
                # Handle date questions
                answer = sorted(remove_duplicates(a_entity), key=len, reverse=True)
                if 'when' in question.lower() or 'what year' in question.lower():
                    for idx in range(len(answer)):
                        if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                            answer[idx] = answer[idx].split('-')[0]
                
                # Determine if double check is needed
                double_check = any([keyword in question.lower() for keyword in 
                                  ['when', 'what year', 'which year', 'where', 'sport', 
                                   "what countr", "language", 'nba finals', 'world series']])
                
                # Calculate metrics
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
                    'id': i,
                    'question': question,
                    'prediction': raw_prediction,
                    'processed_prediction': prediction,
                    'ground_truth': answer
                }
                pred_f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
                
                detail_data = {
                    'id': i,
                    'question': question,
                    'prediction': prediction,
                    'ground_truth': answer,
                    'hit@1': hit1,
                    'hit':hit,
                    'f1': f1_score,
                    'precision': precision_score,
                    'recall': recall_score
                }
                detail_f.write(json.dumps(detail_data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                break
    
    if len(hit_list) == 0:
        print("No valid predictions found!")
        return
    
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
    
    # Print results
    result_str = f"Hit: {avg_hit:.2f}, Hit@1: {avg_hit1:.2f}, Macro F1: {avg_f1:.2f}, Macro Precision: {avg_precision:.2f}, Macro Recall: {avg_recall:.2f}, Exact Match: {num_exact_match:.2f}, Totally Wrong: {num_totally_wrong:.2f}"
    print(result_str)
    print(f"Micro F1: {micro_f1:.4f}, Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}")
    print(f"Total samples: {len(hit_list)}")
    
    # Save final results
    results_file = os.path.join(output_dir, 'final_results.txt')
    with open(results_file, 'w') as f:
        f.write(result_str + '\n')
        f.write(f"Micro F1: {micro_f1:.4f}, Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}\n")
        f.write(f"Total samples: {len(hit_list)}\n")
    
    return {
        'hit':avg_hit,
        'hit@1': avg_hit1,
        'macro_f1': avg_f1,
        'macro_precision': avg_precision,
        'macro_recall': avg_recall,
        'exact_match': num_exact_match,
        'totally_wrong': num_totally_wrong,
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall
    }
    
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
for param in model.parameters():
    param.requires_grad = False  # Freezing the model
    

base_dir = "/home/abdullahm/sourav/notebooks/results/cwq/architecture-v9/rv8-n1000-e30_cosine"
print("Base_dir is:", base_dir)
with open(os.path.join(base_dir,"selected_triplets.json"),"r") as f:
    data = json.load(f) 
print("------------------------------------------------------------------")
print("Length of linearized triplets:",len(data[10]["reranker"]), len(data[122]["reranker"]))
results = evaluate_dataset(
        data,
        base_dir,
        top_k=100
    )
print(f"result saved to {base_dir}")