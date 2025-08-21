# File: examples/bp_online_localLLM/evaluation/runEval.py
"""
Evaluation script for OIG problem in EoH framework.
This script evaluates evolved interdiction heuristics.
"""

import importlib
import numpy as np

def main():
    """
    Main evaluation function called by EoH framework.
    This function imports and tests the evolved heuristic.
    """
    
    try:
        print("Starting OIG heuristic evaluation...")
        
        # Import the evolved heuristic
        heuristic_module = importlib.import_module("heuristic")
        heuristic = importlib.reload(heuristic_module)
        
        # Simple test to verify the heuristic works
        test_prizes = [0, 10, 15, 8, 12]  # depot=0 has prize 0
        test_distances = np.array([
            [0, 5, 10, 8, 12],
            [5, 0, 6, 4, 8],
            [10, 6, 0, 7, 5],
            [8, 4, 7, 0, 9],
            [12, 8, 5, 9, 0]
        ])
        test_depot = 0
        test_budget = 2
        
        # Test the heuristic function
        scores = heuristic.interdiction_score(
            test_prizes, test_distances, test_depot, test_budget
        )
        
        print(f"Test completed successfully.")
        print(f"Prize vector: {test_prizes}")
        print(f"Interdiction scores: {scores}")
        print(f"Depot score: {scores[test_depot]} (should be -inf)")
        
        # Verify basic requirements
        if scores[test_depot] != -float('inf'):
            print("Warning: Depot score is not -inf!")
            return -1000.0
        
        if len(scores) != len(test_prizes):
            print("Warning: Score vector length mismatch!")
            return -1000.0
        
        # Simple fitness calculation (negative sum of non-depot scores for minimization)
        valid_scores = [s for i, s in enumerate(scores) if i != test_depot and s != -float('inf')]
        fitness = -np.mean(valid_scores) if valid_scores else -100.0
        
        print(f"Calculated fitness: {fitness:.4f}")
        
        # Write results to file for EoH
        with open("oig_results.txt", "w") as file:
            file.write(f"Test prizes: {test_prizes}\n")
            file.write(f"Interdiction scores: {scores.tolist()}\n") 
            file.write(f"Fitness: {fitness:.4f}\n")
        
        print("OIG evaluation completed successfully!")
        return fitness
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Write error to results file
        with open("oig_results.txt", "w") as file:
            file.write(f"Evaluation failed: {str(e)}\n")
        
        # Return penalty score for failed evaluation
        return -1000.0

if __name__ == "__main__":
    result = main()
    print(f"Final evaluation result: {result}")