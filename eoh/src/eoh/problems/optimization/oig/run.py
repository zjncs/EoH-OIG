# File: problems/optimization/oig/run.py

import numpy as np
import importlib
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys

class OIGPROBLEM():
    def __init__(self):
        getdata = GetData()
        self.instances, self.lb = getdata.get_instances()
        self.prompts = GetPrompts()

    def simple_orienteering_solver(self, instance, interdicted_nodes):
        """
        Simple greedy solver for the orienteering problem (follower's problem).
        
        Args:
            instance: Dictionary containing the problem instance
            interdicted_nodes: List of nodes that have been interdicted
            
        Returns:
            total_prize: Total prize collected by the follower
        """
        distances = instance['distances']
        prizes = instance['prizes'].copy()
        depot = instance['depot']
        distance_budget = instance['distance_budget']
        n_nodes = instance['n_nodes']
        
        # Remove prizes from interdicted nodes
        for node in interdicted_nodes:
            if node != depot:  # Cannot interdict depot
                prizes[node] = 0
        
        # Greedy orienteering: collect prizes based on prize/distance ratio
        current_node = depot
        total_prize = 0
        total_distance = 0
        visited = {depot}
        
        while True:
            best_node = None
            best_ratio = 0
            
            # Find best unvisited node
            for next_node in range(n_nodes):
                if next_node not in visited and prizes[next_node] > 0:
                    dist_to_next = distances[current_node][next_node]
                    dist_to_depot = distances[next_node][depot]
                    
                    # Check if we can visit this node and return to depot
                    if total_distance + dist_to_next + dist_to_depot <= distance_budget:
                        ratio = prizes[next_node] / (dist_to_next + 1e-6)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_node = next_node
            
            if best_node is None:
                break
                
            # Move to best node
            total_distance += distances[current_node][best_node]
            total_prize += prizes[best_node]
            visited.add(best_node)
            current_node = best_node
        
        return total_prize

    def evaluate_interdiction(self, instance, alg, interdiction_budget):
        """
        Evaluate an interdiction strategy on a single instance.
        
        Args:
            instance: Problem instance
            alg: Algorithm module containing interdiction_score function
            interdiction_budget: Number of nodes that can be interdicted
            
        Returns:
            follower_objective: Prize collected by follower after interdiction
        """
        # Get interdiction scores from the evolved heuristic
        scores = alg.interdiction_score(
            instance['prizes'],
            instance['distances'], 
            instance['depot'],
            interdiction_budget
        )
        
        # Select top-k nodes to interdict (excluding depot)
        valid_indices = [i for i in range(len(scores)) if scores[i] != -float('inf')]
        sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
        interdicted_nodes = sorted_indices[:interdiction_budget]
        
        # Solve follower's orienteering problem
        follower_objective = self.simple_orienteering_solver(instance, interdicted_nodes)
        
        return follower_objective

    def evaluateGreedy(self, alg) -> float:
        """
        Evaluate heuristic function on a set of OIG instances.
        Similar to BPONLINE.evaluateGreedy but for interdiction problem.
        """
        total_fitness = 0
        dataset_count = 0
        
        for name, dataset in self.instances.items():
            follower_objectives = []
            
            for _, instance in dataset.items():
                # Get interdiction budget from instance (or use default)
                interdiction_budget = instance.get('interdiction_budget', 3)
                
                # Evaluate interdiction strategy
                follower_obj = self.evaluate_interdiction(instance, alg, interdiction_budget)
                follower_objectives.append(follower_obj)
            
            # Calculate average follower objective for this dataset
            avg_follower_obj = np.mean(follower_objectives)
            
            # Calculate fitness (goal: minimize follower's objective)
            # Use the same fitness calculation pattern as bp_online
            if self.lb[name] > 0:
                fitness = (self.lb[name] - avg_follower_obj) / self.lb[name]
            else:
                fitness = -avg_follower_obj / 100.0  # Normalize for stability
            
            total_fitness += fitness
            dataset_count += 1
        
        # Return average fitness across all datasets
        return total_fitness / dataset_count if dataset_count > 0 else 0

    def evaluate(self, code_string):
        """
        Evaluate a code string containing an interdiction heuristic.
        This method is called by the EoH framework during evolution.
        """
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Evaluate the heuristic
                fitness = self.evaluateGreedy(heuristic_module)

                return fitness
                
        except Exception as e:
            # Return penalty for failed evaluation
            return None