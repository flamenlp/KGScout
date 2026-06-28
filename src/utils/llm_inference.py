"""
LLM inference utilities for answer generation.

This module provides functions for LLM prompting and inference, adapted from lama-inference.py.
Supports Llama-3.1-8b, Qwen, and DeepSeek models.
"""

import json
import re
import torch
from typing import List, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


# In-context learning examples (from lama-inference.py)
ICL_USER_PROMPT1 = """
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

ICL_ASS_PROMPT1 = """Answer in JSON format:
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

ICL_USER_PROMPT2 = """
Triplets: 
- Steve Bisciotti, sports professional sports team owner s, Baltimore Ravens
- Steve Bisciotti, sports sports team owner teams owned, Baltimore Ravens
- Steve Bisciotti, organization organization founder organizations founded, Allegis Group

Question:
Who is the coach of the team owned by Steve Bisciotti?"""

ICL_ASS_PROMPT2 = """Answer in JSON format:
{"ans": ["answer not available"]}

Reason:
Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.
"""


# def format_prompt_v1(question: str, triplets: List[str], topk: int = None) -> str:
#     """
#     Format LLM prompt with question and selected triplets (ORIGINAL - v1).
#     
#     This function creates a prompt following the format used in lama-inference.py,
#     including in-context learning examples and instructions.
#     
#     Args:
#         question: Question text
#         triplets: List of linearized triplet strings
#         topk: Optional limit on number of triplets to include (default: use all)
#     
#     Returns:
#         Formatted prompt string for LLM inference
#     """
#     # Limit triplets if topk specified
#     if topk is not None:
#         triplets = triplets[:topk]
#     
#     # Format triplets as bullet list
#     triplet_text = "\n".join([f"- {triplet}" for triplet in triplets])
#     
#     # Define answer formats
#     na_format = '{"ans": ["answer not available"]}'
#     ans_format = '{"ans":["your answer 1","your answer 2"]}'
#     
#     # Create user query
#     user_query = f"""
# Linearized Triplets: 
# {triplet_text}
# 
# Question:
# {question}
# 
# Let's think step by step."""
#     
#     # Create full prompt with system message and ICL examples
#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a knowledge graph question answering system. Given a question and relevant linearized knowledge triplets, provide correct answers in JSON format supported by the triplets and provide a brief reason for the answer based on the triplets.
# 
# Instructions:
# 1> Provide your final answer in JSON format using this placeholder: {ans_format}. If there is insufficient information to answer the question, return {na_format}.
# 2> Ensure answer does not contain duplicate entries.
# 3> Keep your reasoning brief and focused.
# 4> Your answer must not contradict any information presented in the provided triplets.
# 5> If the answer is directly supported by the triplets, use the triplets to justify your answer.
# 6> If the answer is not explicitly found in the triplets, you may use your own factual knowledgebut only if it is consistent with the information in the triplets.
# 
# #Example 1:
# {ICL_USER_PROMPT1}
# #Answer:
# {ICL_ASS_PROMPT1}
# 
# #Example 2:
# {ICL_USER_PROMPT2}
# #Answer:
# {ICL_ASS_PROMPT2}
# 
# Now consider the below Triplets and answer the Question carefully
# <|start_header_id|>user<|end_header_id|>
# {user_query}
# #Answer:
# <|start_header_id|>assistant<|end_header_id|>
# """
#     
#     return prompt


def format_prompt(question: str, triplets: List[str], topk: int = None, q_entity: List[str] = None) -> str:
    """
    Format LLM prompt with question and selected triplets (v3 - Concise, no external knowledge).
    
    Simplified prompt that:
      - Explicitly provides the question entity for anchoring
      - Forbids external knowledge usage
      - Keeps instructions concise for 8B model capacity
      - Uses answer-first format for reliable JSON extraction
    
    Args:
        question: Question text
        triplets: List of linearized triplet strings (space-separated format)
        topk: Optional limit on number of triplets to include (default: use all)
        q_entity: Optional list of question entity strings to help LLM anchor
    
    Returns:
        Formatted prompt string for LLM inference
    
    Requirements:
        - 9.1: Use the prompting logic from lama-inference.py for all LLM models
        - 9.2: Format prompts with question text and selected triplets when generating answers
    """
    # Limit triplets if topk specified
    if topk is not None:
        triplets = triplets[:topk]
    
    # Format triplets as bullet list
    triplet_text = "\n".join([f"- {triplet}" for triplet in triplets])
    
    # Define answer formats
    na_format = '{"ans": ["answer not available"]}'
    ans_format = '{"ans": ["answer1", "answer2"]}'
    
    # Format question entity info
    entity_text = ""
    if q_entity:
        label = "Question Entities" if len(q_entity) > 1 else "Question Entity"
        entity_text = f"{label}: {', '.join(q_entity)}\n\n"
    
    # Create user query
    user_query = f"""
{entity_text}Triplets:
{triplet_text}

Question:
{question}

Let's think step by step."""
    
    # ICL Example 1 - successful multi-hop reasoning (comma-separated triplets)
    icl_user_1 = """
Question Entity: Lou Seal

Triplets:
- Lou Seal, sports mascot team, San Francisco Giants
- San Francisco Giants, sports sports team championships, 2012 World Series
- San Francisco Giants, sports sports team championships, 2010 World Series
- San Francisco Giants, sports sports team championships, 2014 World Series
- Crazy Crab, sports mascot team, San Francisco Giants
- New York Yankees, sports sports team championships, 2009 World Series

Question:
What year did the team with mascot named Lou Seal win the World Series?

Let's think step by step."""

    icl_assistant_1 = """{"ans": ["2014", "2012", "2010"]}

Reason: Lou Seal -> San Francisco Giants (via sports mascot team). San Francisco Giants -> 2012 World Series, 2010 World Series, 2014 World Series (via sports sports team championships)."""

    # ICL Example 2 - answer not available
    icl_user_2 = """
Question Entity: Steve Bisciotti

Triplets:
- Steve Bisciotti, sports professional sports team owner s, Baltimore Ravens
- Steve Bisciotti, sports sports team owner teams owned, Baltimore Ravens
- Steve Bisciotti, organization organization founder organizations founded, Allegis Group

Question:
Who is the coach of the team owned by Steve Bisciotti?

Let's think step by step."""

    icl_assistant_2 = """{"ans": ["answer not available"]}

Reason: Steve Bisciotti owns Baltimore Ravens, but no triplet contains a coaching relation for Baltimore Ravens."""

    # Create full prompt with system message and ICL examples
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a knowledge graph question answering system. Given a question, question entities, and relevant knowledge triplets, answer the question using ONLY the provided triplets.

Rules:
1> Answer ONLY from the triplets. Do NOT use external knowledge.
2> Start from the Question Entity, find triplets containing it, then follow connections to other entities until you find the answer.
3> Provide answer in JSON format: {ans_format}. If the answer is not in the triplets, return {na_format}.
4> Keep reasoning brief. No duplicate answers.

#Example 1:
<|start_header_id|>user<|end_header_id|>
{icl_user_1}
<|start_header_id|>assistant<|end_header_id|>
{icl_assistant_1}

#Example 2:
<|start_header_id|>user<|end_header_id|>
{icl_user_2}
<|start_header_id|>assistant<|end_header_id|>
{icl_assistant_2}

#Now answer:
<|start_header_id|>user<|end_header_id|>
{user_query}
<|start_header_id|>assistant<|end_header_id|>
"""
    
    return prompt


def run_llm_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1
) -> str:
    """
    Run LLM inference to generate answer.
    
    Args:
        model: Loaded LLM model
        tokenizer: Loaded tokenizer
        prompt: Formatted prompt string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated answer text
    
    Requirements:
        - 9.3: Support Llama-3.1-8b, Qwen, and DeepSeek model inference
        - 9.4: Log the error and continue with remaining questions when LLM inference fails
        - 9.6: Handle special characters and encoding issues in LLM responses
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response (skip input tokens)
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
        
    except Exception as e:
        # Log error and return empty response
        print(f"Warning: LLM inference failed with error: {str(e)}")
        return ""


def load_llm_model(
    model_name: str,
    device: str = "cuda"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load LLM model and tokenizer.
    
    Args:
        model_name: Model identifier ('llama', 'qwen', 'deepseek')
        device: Target device
    
    Returns:
        Tuple of (model, tokenizer)
    
    Requirements:
        - 9.3: Support Llama-3.1-8b, Qwen, and DeepSeek model inference
    """
    # Map model names to HuggingFace identifiers
    model_map = {
        'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'qwen': 'Qwen/Qwen-7B-Chat',
        'deepseek': 'deepseek-ai/deepseek-llm-7b-chat'
    }
    
    if model_name not in model_map:
        raise ValueError(
            f"Invalid model name: '{model_name}'. "
            f"Expected one of: {list(model_map.keys())}"
        )
    
    model_id = model_map[model_name]
    
    print(f"Loading LLM model: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded successfully on device: {model.device}")
    
    return model, tokenizer
