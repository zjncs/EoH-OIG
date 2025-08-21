# File: problems/optimization/oig/get_instance.py

import numpy as np
import pickle
import os

class GetData():
    def __init__(self) -> None:
        self.datasets = {}
        # Generate synthetic OIG test instances
        self._generate_test_instances()

    def _generate_test_instances(self):
        """Generate synthetic OIG instances for testing."""
        # Create different instance sizes similar to other problems
        configurations = [
            {'name': 'OIG Small', 'n_nodes': 15, 'instances': 5},
            {'name': 'OIG Medium', 'n_nodes': 25, 'instances': 5},
            {'name': 'OIG Large', 'n_nodes': 35, 'instances': 3}
        ]
        
        for config in configurations:
            dataset_name = config['name']
            n_nodes = config['n_nodes']
            n_instances = config['instances']
            
            self.datasets[dataset_name] = {}
            
            for i in range(n_instances):
                instance_name = f"test_{i}"
                instance = self._generate_single_instance(n_nodes, seed=i*100 + n_nodes)
                self.datasets[dataset_name][instance_name] = instance

    def _generate_single_instance(self, n_nodes, seed=42):
        """Generate a single OIG instance."""
        np.random.seed(seed)
        
        # Generate random node coordinates in [0, 100] x [0, 100]
        nodes = np.random.uniform(0, 100, (n_nodes, 2))
        
        # Calculate Euclidean distance matrix
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    distances[i][j] = np.sqrt(np.sum((nodes[i] - nodes[j])**2))
        
        # Generate prizes (depot has prize 0, others have random prizes)
        prizes = [0]  # depot prize
        for i in range(1, n_nodes):
            # Generate prizes using a deterministic but varied formula
            prize = 1 + (7141 * i + 73) % 20  # Range 1-20
            prizes.append(prize)
        
        # Set depot (always node 0)
        depot = 0
        
        # Calculate reasonable distance budget
        avg_distance = np.mean(distances[distances > 0])
        distance_budget = avg_distance * 2.5  # Allow reasonable touring
        
        # Default interdiction budget
        interdiction_budget = min(3, max(1, n_nodes // 5))  # About 20% of nodes
        
        return {
            'nodes': nodes.tolist(),
            'distances': distances,
            'prizes': prizes,
            'depot': depot,
            'distance_budget': distance_budget,
            'interdiction_budget': interdiction_budget,
            'n_nodes': n_nodes,
            'type': 'synthetic'
        }

    def calculate_upper_bound(self, instance):
        """
        Calculate upper bound for follower's objective (no interdiction scenario).
        This represents the best the follower can do without any interdiction.
        """
        distances = instance['distances']
        prizes = instance['prizes']
        depot = instance['depot']
        distance_budget = instance['distance_budget']
        n_nodes = instance['n_nodes']
        
        # Use greedy orienteering solver to get upper bound
        current_node = depot
        total_prize = 0
        total_distance = 0
        visited = {depot}
        
        while True:
            best_node = None
            best_ratio = 0
            
            for next_node in range(n_nodes):
                if next_node not in visited and prizes[next_node] > 0:
                    dist_to_next = distances[current_node][next_node]
                    dist_to_depot = distances[next_node][depot]
                    
                    if total_distance + dist_to_next + dist_to_depot <= distance_budget:
                        ratio = prizes[next_node] / (dist_to_next + 1e-6)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_node = next_node
            
            if best_node is None:
                break
                
            total_distance += distances[current_node][best_node]
            total_prize += prizes[best_node]
            visited.add(best_node)
            current_node = best_node
        
        return total_prize

    def calculate_dataset_bounds(self, instances):
        """
        Calculate performance bounds for a dataset.
        Returns average upper bound (worst case for leader).
        """
        upper_bounds = []
        for instance_name, instance in instances.items():
            upper_bound = self.calculate_upper_bound(instance)
            upper_bounds.append(upper_bound)
        
        return np.mean(upper_bounds)

    def get_instances(self):
        """
        Get instances and calculate performance bounds.
        Returns instances and upper bounds (representing worst case for leader).
        """
        # Calculate bounds for all datasets
        bounds = {}
        for name, dataset in self.datasets.items():
            bounds[name] = self.calculate_dataset_bounds(dataset)
        
        print("OIG datasets loaded:")
        for name, bound in bounds.items():
            dataset_size = len(self.datasets[name])
            avg_nodes = np.mean([inst['n_nodes'] for inst in self.datasets[name].values()])
            print(f"  {name}: {dataset_size} instances, avg {avg_nodes:.0f} nodes, upper bound = {bound:.2f}")
        
        return self.datasets, bounds