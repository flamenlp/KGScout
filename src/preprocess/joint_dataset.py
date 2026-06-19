"""
Joint training dataset with Personalized PageRank (PPR) features.

This module implements JointTrainingDatasetv3PPR which precomputes PPR scores
for knowledge graph triplets using NetworkX.
"""

import torch
from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Optional
import logging

# Configure logging for error tracking
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class JointTrainingDatasetv3PPR(Dataset):
    """
    Dataset that computes PPR (Personalized PageRank) graph features.
    
    For each sample:
    - Builds directed graph from triplets
    - Computes PPR scores personalized to question entities
    - Stores precomputed features for efficient training
    
    Handles edge cases:
    - Empty question entities: returns zero features
    - Empty triplets: returns zero features
    - Invalid triplets (not 3-tuples): skips and logs
    """
    
    def __init__(self, train_data: List[Dict], device: str = 'cpu'):
        """
        Precompute graph features for all samples.
        
        Args:
            train_data: List of dicts with keys:
                - question: Question text
                - q_entity: List of question entities
                - a_entity: List of answer entities
                - topk_rel_data: List of (score, triplet) tuples
                - question_embedding: Question embedding tensor
                - topk_linearized_triplets: Linearized triplet strings
                - topk_linearized_triplet_embeddings: Triplet embeddings
                - topK_rel_embeddings: Relation embeddings
                - is_empty: Whether sample has empty paths
                - answer: Answer text
            device: Device to store tensors on ('cpu' or 'cuda')
        """
        self.device = device
        self.precomputed_data = []
        self.skipped_samples = 0
        self.skipped_reasons = {
            'missing_fields': 0,
            'invalid_triplet_format': 0,
            'ppr_computation_failure': 0,
            'other_errors': 0
        }
        
        for idx, entry in enumerate(tqdm(train_data, total=len(train_data), desc="Precomputing PPR features")):
            try:
                # Validate required fields
                required_fields = [
                    "question", "q_entity", "a_entity", "topk_rel_data",
                    "question_embedding", "topk_linearized_triplets",
                    "topk_linearized_triplet_embeddings", "topK_rel_embeddings",
                    "is_empty", "answer"
                ]
                
                missing_fields = [field for field in required_fields if field not in entry]
                if missing_fields:
                    raise KeyError(f"Missing required fields: {missing_fields}")
                
                # Normalize question entities to lowercase
                q_entity = [e.lower() for e in entry["q_entity"]]
                triplets = [t[1] for t in entry["topk_rel_data"]]
                
                # Handle empty samples
                if len(q_entity) == 0 or len(triplets) == 0:
                    graph_feats = torch.zeros((1, 2), dtype=torch.float32)
                else:
                    # Validate triplet format (must be 3-tuples)
                    for i, triplet in enumerate(triplets):
                        if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
                            raise ValueError(
                                f"Invalid triplet format at index {i}: {triplet}. "
                                f"Expected 3-tuple (subject, relation, object), got {type(triplet)} with length {len(triplet) if hasattr(triplet, '__len__') else 'N/A'}"
                            )
                    
                    # Build directed graph from triplets
                    G = nx.DiGraph()
                    for (s, r, o) in triplets:
                        G.add_edge(s.lower(), o.lower(), relation=r.lower())
                    
                    # Compute personalized PageRank scores
                    try:
                        personalization = {n: (1.0 if n in q_entity else 0.0) for n in G.nodes()}
                        
                        # Check if personalization is valid (at least one non-zero value)
                        if sum(personalization.values()) == 0:
                            logger.warning(
                                f"Sample {idx}: No question entities found in graph. "
                                f"Using zero features."
                            )
                            graph_feats = torch.zeros((len(triplets), 2), dtype=torch.float32)
                        else:
                            ppr_scores = nx.pagerank(
                                G,
                                alpha=0.85,
                                personalization=personalization,
                                max_iter=100,
                                tol=1e-05
                            )
                            
                            # Extract PPR scores for each triplet's subject and object
                            graph_feats = []
                            for (s, r, o) in triplets:
                                s_, o_ = s.lower(), o.lower()
                                ppr_s = ppr_scores.get(s_, 0.0)
                                ppr_o = ppr_scores.get(o_, 0.0)
                                graph_feats.append([ppr_s, ppr_o])
                            
                            graph_feats = torch.tensor(graph_feats, dtype=torch.float32)
                            
                    except nx.PowerIterationFailedConvergence as e:
                        # PPR computation failed to converge - use zero features
                        logger.warning(
                            f"Sample {idx}: PPR computation failed to converge after {e.iterations} iterations. "
                            f"Using zero features. Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
                        )
                        graph_feats = torch.zeros((len(triplets), 2), dtype=torch.float32)
                        self.skipped_samples += 1
                        self.skipped_reasons['ppr_computation_failure'] += 1
                    except Exception as e:
                        # Other PPR computation errors
                        logger.warning(
                            f"Sample {idx}: PPR computation failed with error: {e}. "
                            f"Using zero features."
                        )
                        graph_feats = torch.zeros((len(triplets), 2), dtype=torch.float32)
                        self.skipped_samples += 1
                        self.skipped_reasons['ppr_computation_failure'] += 1
                
                # Store precomputed sample
                self.precomputed_data.append({
                    "question": entry["question"],
                    "is_empty": entry["is_empty"],
                    "q_entity": entry["q_entity"],
                    "a_entity": entry["a_entity"],
                    "answer": entry["answer"],
                    "question_embedding": entry["question_embedding"],
                    "topk_linearized_triplets": entry["topk_linearized_triplets"],
                    "topk_linearized_triplet_embeddings": entry["topk_linearized_triplet_embeddings"],
                    "topk_rel_data": entry["topk_rel_data"],
                    "topK_rel_embeddings": entry["topK_rel_embeddings"],
                    "graph_features": graph_feats.to(self.device)
                })
                
            except KeyError as e:
                # Missing required fields - skip sample
                logger.warning(f"Sample {idx}: {e}. Skipping sample.")
                self.skipped_samples += 1
                self.skipped_reasons['missing_fields'] += 1
                
            except ValueError as e:
                # Invalid triplet format - skip sample
                logger.warning(f"Sample {idx}: {e}. Skipping sample.")
                self.skipped_samples += 1
                self.skipped_reasons['invalid_triplet_format'] += 1
                
            except Exception as e:
                # Unexpected error - skip sample
                logger.warning(
                    f"Sample {idx}: Unexpected error during preprocessing: {e}. "
                    f"Skipping sample."
                )
                self.skipped_samples += 1
                self.skipped_reasons['other_errors'] += 1
        
        # Report skipped samples summary
        if self.skipped_samples > 0:
            logger.warning(
                f"\n{'='*60}\n"
                f"Preprocessing Summary:\n"
                f"  Total samples processed: {len(train_data)}\n"
                f"  Successfully processed: {len(self.precomputed_data)}\n"
                f"  Skipped samples: {self.skipped_samples}\n"
                f"  Breakdown:\n"
                f"    - Missing required fields: {self.skipped_reasons['missing_fields']}\n"
                f"    - Invalid triplet format: {self.skipped_reasons['invalid_triplet_format']}\n"
                f"    - PPR computation failures: {self.skipped_reasons['ppr_computation_failure']}\n"
                f"    - Other errors: {self.skipped_reasons['other_errors']}\n"
                f"{'='*60}"
            )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.precomputed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return precomputed sample with graph features.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing all sample data including precomputed graph_features
        """
        return self.precomputed_data[idx]
