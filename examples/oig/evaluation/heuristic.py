# File: examples/bp_online_localLLM/evaluation/heuristic.py

import numpy as np

def interdiction_score(prizes, distances, depot, interdiction_budget):
    """
    Basic interdiction heuristic - prioritize highest prizes.
    This file will be evolved by EoH to find better strategies.
    
    Args:
        prizes: List of prizes at each node (depot has prize 0)
        distances: 2D numpy array of distances between nodes
        depot: Index of depot node (usually 0)  
        interdiction_budget: Max number of nodes that can be interdicted
        
    Returns:
        scores: Numpy array where scores[i] = value of interdicting node i
    """
    n_nodes = len(prizes)
    scores = np.array(prizes, dtype=float)
    
    # Cannot interdict depot
    scores[depot] = -float('inf')
    
    return scores