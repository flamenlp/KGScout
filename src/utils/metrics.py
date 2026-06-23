"""
Evaluation metrics for KGQA system.

This module provides functions for computing evaluation metrics:
- Hit and Hit@1 scores
- F1, Precision, and Recall scores
- Answer coverage
- Path coverage
- Answer normalization
"""

import re
import string
import json
import networkx as nx
from typing import List, Tuple, Dict
from copy import deepcopy


def normalize_answer(answer: str) -> str:
    """
    Normalize answer text for comparison.
    
    Normalization steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove special tokens (<pad>)
    5. Collapse whitespace
    
    Args:
        answer: Answer text to normalize
    
    Returns:
        Normalized answer text
    
    Requirements:
        - 9.5: Compute answer quality metrics by comparing LLM output against ground truth answers
    """
    # Convert to lowercase
    s = answer.lower()
    
    # Remove punctuation
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    
    # Remove special tokens
    s = re.sub(r"\b(<pad>)\b", " ", s)
    
    # Collapse whitespace
    s = " ".join(s.split())
    
    return s


def match_answer(predicted: str, ground_truth: str) -> bool:
    """
    Check if ground_truth is contained in predicted after normalization.
    
    Args:
        predicted: Predicted answer text
        ground_truth: Ground truth answer text
    
    Returns:
        True if ground_truth is contained in predicted
    """
    predicted_norm = normalize_answer(predicted)
    ground_truth_norm = normalize_answer(ground_truth)
    return ground_truth_norm in predicted_norm


def remove_duplicates(input_list: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def extract_predictions_from_response(response: str) -> List[str]:
    """
    Extract predictions from LLM response.
    
    Tries to parse JSON format first, falls back to heuristics if JSON parsing fails.
    
    Args:
        response: Raw LLM response text
    
    Returns:
        List of predicted answer strings
    """
    # Try to extract JSON
    pattern = r'\{[^{}]*\}'
    json_match = re.search(pattern, response)
    
    if not json_match:
        # No JSON found - try to extract from "ans": [ format
        if '{"ans": [' in response:
            x = response.split('{"ans": [')[-1]
            entities = [ele.strip().strip('"').lower() for ele in x.split(",")]
            return remove_duplicates(entities)
        else:
            # Fall back to using the entire response
            return [response.strip()]
    
    # Try to parse JSON
    json_str = json_match.group(0)
    try:
        data = json.loads(json_str)
        predictions = remove_duplicates(data["ans"])
        return predictions
    except Exception:
        # JSON parsing failed - use entire response
        return [response.strip()]


def compute_precision(
    prediction: List[str],
    ground_truth: List[str],
    double_check: bool = False
) -> Tuple[float, int, int]:
    """
    Calculate precision score.
    
    Precision = (number of correct predictions) / (total predictions)
    
    Args:
        prediction: List of predicted answers
        ground_truth: List of ground truth answers
        double_check: Whether to use additional matching heuristics
    
    Returns:
        Tuple of (precision_score, matched_count, total_predictions)
    
    Requirements:
        - 9.5: Compute answer quality metrics by comparing LLM output against ground truth answers
    """
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    num_pred = len(prediction)
    
    if num_pred == 0:
        return 0.0, 0, 0
    
    matched = 0
    for gt in ground_truth:
        for pred in prediction:
            if match_answer(pred, gt):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                # Additional heuristic: check if answer is after "ans:" prefix
                if match_answer(gt, pred.split('ans:')[-1].strip()) or match_answer(gt, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    
    precision = matched / num_pred
    return precision, matched, num_pred


def compute_recall(
    prediction: List[str],
    ground_truth: List[str],
    double_check: bool = False
) -> Tuple[float, int, int]:
    """
    Calculate recall score.
    
    Recall = (number of correct predictions) / (total ground truth answers)
    
    Args:
        prediction: List of predicted answers
        ground_truth: List of ground truth answers
        double_check: Whether to use additional matching heuristics
    
    Returns:
        Tuple of (recall_score, matched_count, total_ground_truth)
    
    Requirements:
        - 9.5: Compute answer quality metrics by comparing LLM output against ground truth answers
    """
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    
    matched = 0
    for gt in ground_truth:
        for pred in prediction:
            if match_answer(pred, gt):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                # Additional heuristic: check if answer is after "ans:" prefix
                if match_answer(gt, pred.split('ans:')[-1].strip()) or match_answer(gt, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    
    recall = matched / len(ground_truth) if len(ground_truth) > 0 else 0.0
    return recall, matched, len(ground_truth)


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Precision score
        recall: Recall score
    
    Returns:
        F1 score
    
    Requirements:
        - 9.5: Compute answer quality metrics by comparing LLM output against ground truth answers
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hit_score(
    prediction: List[str],
    ground_truth: List[str],
    double_check: bool = False
) -> float:
    """
    Calculate Hit score (whether any prediction matches any ground truth).
    
    Args:
        prediction: List of predicted answers
        ground_truth: List of ground truth answers
        double_check: Whether to use additional matching heuristics
    
    Returns:
        1.0 if any prediction matches, 0.0 otherwise
    
    Requirements:
        - 9.5: Compute answer quality metrics by comparing LLM output against ground truth answers
    """
    if len(prediction) == 0:
        return 0.0
    
    for gt in ground_truth:
        for pred in prediction:
            if match_answer(pred, gt):
                return 1.0
            elif double_check and match_answer(gt, pred.strip()):
                return 1.0
    
    return 0.0


def compute_hit_at_1(
    prediction: List[str],
    ground_truth: List[str],
    double_check: bool = False
) -> float:
    """
    Calculate Hit@1 score (whether first prediction matches any ground truth).
    
    Args:
        prediction: List of predicted answers
        ground_truth: List of ground truth answers
        double_check: Whether to use additional matching heuristics
    
    Returns:
        1.0 if first prediction matches, 0.0 otherwise
    
    Requirements:
        - 9.5: Compute answer quality metrics by comparing LLM output against ground truth answers
    """
    if len(prediction) == 0:
        return 0.0
    
    first_pred = prediction[0]
    for gt in ground_truth:
        if match_answer(first_pred, gt):
            return 1.0
        elif double_check and match_answer(gt, first_pred.strip()):
            return 1.0
    
    return 0.0


def compute_answer_coverage(
    triplets: List[Tuple[str, str, str]],
    answer_entities: List[str]
) -> bool:
    """
    Compute answer coverage metric.
    
    Answer coverage is True if any answer entity appears in the selected triplets
    (either as subject or object).
    
    Args:
        triplets: List of (subject, relation, object) tuples
        answer_entities: List of answer entity strings
    
    Returns:
        True if any answer entity is found in the triplets, False otherwise
    
    Requirements:
        - 3.3: Check if answer entities exist in the selected triplets for each question
    """
    if not answer_entities:
        return False
    
    # Check if any answer entity is present in any triplet
    return any(
        ent.lower() in {s.lower(), o.lower()}
        for ent in answer_entities
        for s, _, o in triplets
    )


def compute_path_coverage(
    triplets: List[Tuple[str, str, str]],
    question_entities: List[str],
    answer_entities: List[str]
) -> bool:
    """
    Compute path coverage metric.
    
    Path coverage is True if there exists a path in the triplet graph
    from at least one question entity to at least one answer entity,
    using an undirected graph (matching notebook implementation).
    
    Args:
        triplets: List of (subject, relation, object) tuples
        question_entities: List of question entity strings
        answer_entities: List of answer entity strings
    
    Returns:
        True if a reasoning path exists, False otherwise
    
    Requirements:
        - 3.4: Check if a complete Reasoning_Path exists in the selected triplets for each question
    """
    if not triplets or not question_entities or not answer_entities:
        return False
    
    # Build undirected graph from triplets (matches notebook)
    G = nx.Graph()
    for s, r, o in triplets:
        G.add_edge(s.lower(), o.lower(), relation=r.lower())
    
    # Check if path exists between any question entity and any answer entity
    for q_entity in question_entities:
        for a_entity in answer_entities:
            qn, an = q_entity.lower(), a_entity.lower()
            if qn not in G or an not in G:
                continue
            try:
                if nx.has_path(G, qn, an):
                    return True
            except nx.NetworkXError:
                continue
    
    return False


def should_use_double_check(question: str) -> bool:
    """
    Determine if double-check heuristics should be used for a question.
    
    Double-check is used for questions about dates, locations, sports, etc.
    where answers may have multiple valid formats.
    
    Args:
        question: Question text
    
    Returns:
        True if double-check should be used
    """
    keywords = [
        'when', 'what year', 'which year', 'where', 'sport',
        'what countr', 'language', 'nba finals', 'world series'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in keywords)


def preprocess_date_answers(question: str, answers: List[str]) -> List[str]:
    """
    Preprocess answers for date questions.
    
    For date questions, extract just the year from date strings (e.g., "2010-01-15" -> "2010").
    
    Args:
        question: Question text
        answers: List of answer strings
    
    Returns:
        Preprocessed answer list
    """
    question_lower = question.lower()
    if 'when' in question_lower or 'what year' in question_lower:
        processed = []
        for answer in answers:
            if '-' in answer and answer.split('-')[0].isdigit():
                processed.append(answer.split('-')[0])
            else:
                processed.append(answer)
        return processed
    return answers
