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
    Format LLM prompt with question and selected triplets (v4 - Full instructions, Llama-3.1 capable).
    
    Comprehensive prompt with:
      - Detailed instructions leveraging Llama-3.1-8B instruction-following capability
      - Question analysis guidance (generic, no domain-specific types)
      - Multi-hop chaining instruction
      - Conjunction/multi-constraint handling
      - 4 ICL examples: 2-hop forward, 2-hop backward (CVT), conjunction, answer not available
    
    Args:
        question: Question text
        triplets: List of linearized triplet strings (comma-separated format)
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
    
    # ICL Example 1 - 2-hop forward (composition)
    icl_user_1 = """
Question:
What year did the team with mascot named Lou Seal win the World Series?

Question Entity: Lou Seal

Triplets:
- Lou Seal, sports mascot team, San Francisco Giants
- San Francisco Giants, sports sports team championships, 2012 World Series
- San Francisco Giants, sports sports team championships, 2010 World Series
- San Francisco Giants, sports sports team championships, 2014 World Series
- Crazy Crab, sports mascot team, San Francisco Giants
- New York Yankees, sports sports team championships, 2009 World Series

Let's think step by step."""

    icl_assistant_1 = """{"ans": ["2014", "2012", "2010"]}

Reason:
The question asks for the years a team won the World Series, where the team has mascot Lou Seal.
From the triplets, "Lou Seal, sports mascot team, San Francisco Giants" tells us Lou Seal's team is San Francisco Giants.
Then "San Francisco Giants, sports sports team championships, 2012/2010/2014 World Series" — the relation "championships" corresponds to "win the World Series".
Answer: 2014, 2012, 2010."""

    # ICL Example 2 - 2-hop backward (CVT node pattern)
    icl_user_2 = """
Question:
What award was Nina Dobrev nominated for?

Question Entity: Nina Dobrev

Triplets:
- m.0z8n1zp, award award nomination award nominee, Nina Dobrev
- m.0z8n1zp, award award nomination award, Teen Choice Award
- Nina Dobrev, people person nationality, Canada

Let's think step by step."""

    icl_assistant_2 = """{"ans": ["Teen Choice Award"]}

Reason:
The question asks what award Nina Dobrev was nominated for.
From the triplets, Nina Dobrev appears at the end of "m.0z8n1zp, award award nomination award nominee, Nina Dobrev" — the relation "award nominee" matches "nominated for". This connects to intermediate entity m.0z8n1zp.
Then "m.0z8n1zp, award award nomination award, Teen Choice Award" — the relation "award" gives us the award name.
Answer: Teen Choice Award."""

    # ICL Example 3 - Conjunction (multiple constraints)
    icl_user_3 = """
Question:
Which film directed by James Cameron also stars Leonardo DiCaprio?

Question Entities: James Cameron, Leonardo DiCaprio

Triplets:
- Titanic, film film director, James Cameron
- Avatar, film film director, James Cameron
- Titanic, film film actor, Leonardo DiCaprio
- The Revenant, film film actor, Leonardo DiCaprio

Let's think step by step."""

    icl_assistant_3 = """{"ans": ["Titanic"]}

Reason:
The question has two constraints: directed by James Cameron AND stars Leonardo DiCaprio.
From "film director" relation: James Cameron directed Titanic, Avatar.
From "film actor" relation: Leonardo DiCaprio acted in Titanic, The Revenant.
The entity satisfying both constraints is Titanic.
Answer: Titanic."""

    # ICL Example 4 - Answer not available
    icl_user_4 = """
Question:
Who is the coach of the team owned by Steve Bisciotti?

Question Entity: Steve Bisciotti

Triplets:
- Steve Bisciotti, sports professional sports team owner s, Baltimore Ravens
- Steve Bisciotti, sports sports team owner teams owned, Baltimore Ravens
- Steve Bisciotti, organization organization founder organizations founded, Allegis Group

Let's think step by step."""

    icl_assistant_4 = """{"ans": ["answer not available"]}

Reason:
The question asks for the coach of the team owned by Steve Bisciotti.
From the triplets, "Steve Bisciotti, sports sports team owner teams owned, Baltimore Ravens" — so the team is Baltimore Ravens.
However, no triplet contains a coaching relation for Baltimore Ravens. The answer is not available in the provided triplets."""

    # Create full prompt with system message and ICL examples
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a knowledge graph question answering system. Given a question, question entities, and relevant knowledge triplets, provide correct answers in JSON format derived from the provided triplets.

Instructions:
1> First, analyze the question to identify: (a) what is being asked for, (b) what relations or properties the question refers to, and (c) whether the question has multiple constraints.
2> Each triplet is formatted as (subject, relation, object). The Question Entity may appear at the start OR end of a triplet — check both positions.
3> Starting from the Question Entity, find triplets containing it, then follow connections through shared entities across triplets until you reach the answer. You may need to chain through 2-3 triplets.
4> Match the meaning of triplet relations to what the question asks. Only follow relations that are semantically relevant to the question.
5> For questions with multiple conditions, find the entity that satisfies ALL conditions from the triplets.
6> Provide your final answer in JSON format: {ans_format}. If the answer cannot be found in the triplets, return {na_format}.
8> Ensure answers do not contain duplicate entries. Keep reasoning brief.

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

#Example 3:
<|start_header_id|>user<|end_header_id|>
{icl_user_3}
<|start_header_id|>assistant<|end_header_id|>
{icl_assistant_3}

#Example 4:
<|start_header_id|>user<|end_header_id|>
{icl_user_4}
<|start_header_id|>assistant<|end_header_id|>
{icl_assistant_4}

#Now answer:
<|start_header_id|>user<|end_header_id|>
{user_query}
<|start_header_id|>assistant<|end_header_id|>
"""
    
    return prompt


def format_prompt_v5(question: str, triplets: List[str], topk: int = None, q_entity: List[str] = None) -> str:
    """
    Format LLM prompt v5 — Optimized for reducing wrong-entity and answer-not-available failures.
    
    Key changes from v4:
      - Removed "answer not available" ICL example (caused 21% of failures by teaching LLM to give up)
      - Replaced with a harder disambiguation example where multiple entities are candidates
      - Added explicit constraint-filtering ICL example with distractors (addresses 76% wrong-entity failures)
      - Stronger instruction against defaulting to "answer not available"
      - Added scanning guidance for large triplet sets
      - Added instruction to prefer entities that satisfy ALL question constraints
    
    Args:
        question: Question text
        triplets: List of linearized triplet strings (comma-separated format)
        topk: Optional limit on number of triplets to include (default: use all)
        q_entity: Optional list of question entity strings to help LLM anchor
    
    Returns:
        Formatted prompt string for LLM inference
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
    
    # ICL Example 1 - 2-hop forward (composition) — kept from v4
    icl_user_1 = """
Question:
What year did the team with mascot named Lou Seal win the World Series?

Question Entity: Lou Seal

Triplets:
- Lou Seal, sports mascot team, San Francisco Giants
- San Francisco Giants, sports sports team championships, 2012 World Series
- San Francisco Giants, sports sports team championships, 2010 World Series
- San Francisco Giants, sports sports team championships, 2014 World Series
- Crazy Crab, sports mascot team, San Francisco Giants
- New York Yankees, sports sports team championships, 2009 World Series
- San Francisco Giants, sports sports team colors, Blue and White
- San Francisco Giants, sports professional sports team owner s, Bill Gates

Let's think step by step."""

    icl_assistant_1 = """{"ans": ["2014", "2012", "2010"]}

Reason:
The question asks for the years a team won the World Series, where the team has mascot Lou Seal.
Step 1: Find Lou Seal's team. "Lou Seal, sports mascot team, San Francisco Giants" → team is San Francisco Giants.
Step 2: Find championships. "San Francisco Giants, sports sports team championships, 2012/2010/2014 World Series".
Note: New York Yankees also won a World Series but is NOT Lou Seal's team, so we exclude it.
Answer: 2014, 2012, 2010."""

    # ICL Example 2 - 2-hop backward (CVT node pattern) — kept from v4
    icl_user_2 = """
Question:
What award was Nina Dobrev nominated for?

Question Entity: Nina Dobrev

Triplets:
- m.0z8n1zp, award award nomination award nominee, Nina Dobrev
- m.0z8n1zp, award award nomination award, Teen Choice Award
- m.0x7k2p1, award award nomination award nominee, Ian Somerhalder
- m.0x7k2p1, award award nomination award, People's Choice Award
- Nina Dobrev, people person nationality, Canada

Let's think step by step."""

    icl_assistant_2 = """{"ans": ["Teen Choice Award"]}

Reason:
The question asks what award Nina Dobrev was nominated for.
Step 1: Find triplets containing Nina Dobrev. "m.0z8n1zp, award award nomination award nominee, Nina Dobrev" — this connects Nina Dobrev to intermediate node m.0z8n1zp.
Step 2: Follow m.0z8n1zp. "m.0z8n1zp, award award nomination award, Teen Choice Award" — the award is Teen Choice Award.
Note: People's Choice Award is connected to Ian Somerhalder (m.0x7k2p1), not Nina Dobrev. We must only follow paths from the question entity.
Answer: Teen Choice Award."""

    # ICL Example 3 - Conjunction with distractors (harder than v4 — addresses wrong-entity failures)
    icl_user_3 = """
Question:
What group that fought in the Siege of Vicksburg was based in Montgomery?

Question Entities: Siege of Vicksburg, Montgomery

Triplets:
- Confederate States of America, location country capital, Montgomery
- Siege of Vicksburg, base culturalevent event entity involved, Confederate States of America
- Siege of Vicksburg, base culturalevent event entity involved, United States of America
- Siege of Vicksburg, base culturalevent event entity involved, Union
- Siege of Vicksburg, time event included in event, American Civil War
- Montgomery, location location containedby, Alabama
- Alabama, location us state capital, Montgomery
- United States of America, location country administrative divisions, Alabama

Let's think step by step."""

    icl_assistant_3 = """{"ans": ["Confederate States of America"]}

Reason:
The question has TWO constraints: (1) fought in Siege of Vicksburg, AND (2) based in Montgomery.
Step 1 (Constraint 1): Find entities involved in the Siege. From triplets: Confederate States of America, United States of America, and Union all fought.
Step 2 (Constraint 2): Check which of those is "based in Montgomery." "Confederate States of America, location country capital, Montgomery" — the Confederacy's capital was Montgomery.
United States of America and Union have no triplet connecting them to Montgomery as a base.
The ONLY entity satisfying BOTH constraints is Confederate States of America.
Answer: Confederate States of America."""

    # ICL Example 4 - Disambiguation among multiple actor/role candidates
    icl_user_4 = """
Question:
Who played the young version of the main character in Forrest Gump?

Question Entities: Forrest Gump

Triplets:
- m.0jycvw, film performance actor, Tom Hanks
- m.0jycvw, film performance character, Forrest Gump
- m.0jycvw, film performance film, Forrest Gump
- m.02xgww5, film performance actor, Michael Connor Humphreys
- m.02xgww5, film performance character, Young Forrest Gump
- m.02xgww5, film performance film, Forrest Gump
- m.0y54dnx, film performance actor, Kevin Mangan
- m.0y54dnx, film performance character, Jenny's Father
- m.0y54dnx, film performance film, Forrest Gump
- Forrest Gump, film film language, English Language

Let's think step by step."""

    icl_assistant_4 = """{"ans": ["Michael Connor Humphreys"]}

Reason:
The question asks who played the "young version" of the main character (Forrest Gump) in the film.
Multiple actors appear: Tom Hanks, Michael Connor Humphreys, Kevin Mangan.
Step 1: Identify which performance corresponds to the "young" version. "m.02xgww5, film performance character, Young Forrest Gump" — this is the young version.
Step 2: Find the actor. "m.02xgww5, film performance actor, Michael Connor Humphreys".
Note: Tom Hanks plays the adult Forrest Gump, Kevin Mangan plays Jenny's Father — neither matches "young version of main character."
Answer: Michael Connor Humphreys."""

    # Create full prompt with system message and ICL examples
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a knowledge graph question answering system. Given a question, question entities, and relevant knowledge triplets, provide correct answers in JSON format derived from the provided triplets.

Instructions:
1> First, analyze the question to identify: (a) what is being asked for, (b) what constraints the question specifies, and (c) what relations would connect the question entity to the answer.
2> Each triplet is (subject, relation, object). The Question Entity may appear as subject OR object — check both positions.
3> Starting from the Question Entity, find triplets containing it, then follow connections through shared entities until you reach the answer. You may need to chain through 2-3 triplets via intermediate nodes (e.g., m.xxxxx CVT nodes).
4> CRITICAL: When multiple candidate entities appear, verify each one against ALL constraints in the question. Only select entities that satisfy every condition. Reject candidates that match only some constraints.
5> Match the semantic meaning of relations to what the question asks. "film performance character" relates to "who played", "location country capital" relates to "based in", etc.
6> The triplets are ordered by relevance. Focus on the first 30 triplets for the main reasoning path, but scan further if needed.
7> Provide your answer in JSON format: {ans_format}. Only return {na_format} if you are certain that NO triplet contains any entity that could answer the question. When in doubt, provide your best answer from the triplets.
8> Ensure answers do not contain duplicate entries. Keep reasoning brief and structured.

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

#Example 3:
<|start_header_id|>user<|end_header_id|>
{icl_user_3}
<|start_header_id|>assistant<|end_header_id|>
{icl_assistant_3}

#Example 4:
<|start_header_id|>user<|end_header_id|>
{icl_user_4}
<|start_header_id|>assistant<|end_header_id|>
{icl_assistant_4}

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
