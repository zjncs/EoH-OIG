# generate_oig_instances.py
# Generate and save OIG test instances in pickle format

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Any

class InstanceGenerator:
    def __init__(self):
        pass
    
    def generate_grid_instance(self, grid_size: int, seed: int = None) -> Dict[str, Any]:
        """Generate instance on a grid (similar to paper's setup)."""
        if seed:
            np.random.seed(seed)
        
        # Create grid coordinates
        nodes = []
        for i in range(grid_size):
            for j in range(grid_size):
                nodes.append((i * 10.0, j * 10.0))
        
        n_nodes = len(nodes)
        
        # Generate prizes (depot has prize 0)
        prizes = [0]  # depot
        for i in range(1, n_nodes):
            # Use paper's formula or random prizes
            prize = 1 + (7141 * i + 73) % 100
            prizes.append(prize)
        
        # Distance budget as fraction of grid diagonal
        diagonal = np.sqrt(2) * (grid_size - 1) * 10
        distance_budget = 0.6 * diagonal
        
        return {
            'nodes': nodes,
            'prizes': prizes,
            'depot': 0,
            'distance_budget': distance_budget,
            'n_nodes': n_nodes,
            'type': 'grid'
        }
    
    def generate_random_instance(self, n_nodes: int, seed: int = None) -> Dict[str, Any]:
        """Generate random geometric instance."""
        if seed:
            np.random.seed(seed)
        
        # Random coordinates in [0, 100] x [0, 100]
        nodes = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) 
                for _ in range(n_nodes)]
        
        # Unit prizes except depot
        prizes = [0] + [1] * (n_nodes - 1)
        
        # Distance budget based on convex hull perimeter approximation
        coords = np.array(nodes)
        diameter = np.max(np.linalg.norm(coords[:, None] - coords[None, :], axis=2))
        distance_budget = 0.5 * diameter
        
        return {
            'nodes': nodes,
            'prizes': prizes,
            'depot': 0,
            'distance_budget': distance_budget,
            'n_nodes': n_nodes,
            'type': 'random'
        }
    
    def generate_dataset(self, instance_type: str, sizes: List[int], 
                        instances_per_size: int = 5) -> Dict[int, List[Dict]]:
        """Generate complete dataset."""
        dataset = {}
        
        for size in sizes:
            instances = []
            for i in range(instances_per_size):
                if instance_type == 'grid':
                    # For grid, size is grid dimension
                    grid_dim = int(np.sqrt(size))
                    instance = self.generate_grid_instance(grid_dim, seed=i*100 + size)
                else:
                    instance = self.generate_random_instance(size, seed=i*100 + size)
                
                instances.append(instance)
            
            dataset[size] = instances
        
        return dataset

def main():
    """Generate and save test datasets."""
    generator = InstanceGenerator()
    
    # Create testingdata directory
    os.makedirs('./testingdata', exist_ok=True)
    
    # Generate different dataset sizes
    datasets = {
        'small': [16, 25, 36],    # Grid sizes: 4x4, 5x5, 6x6
        'medium': [49, 64, 81],   # Grid sizes: 7x7, 8x8, 9x9  
        'large': [100, 121, 144]  # Grid sizes: 10x10, 11x11, 12x12
    }
    
    for size_category, sizes in datasets.items():
        print(f"Generating {size_category} instances...")
        
        # Generate grid instances
        grid_dataset = generator.generate_dataset('grid', sizes)
        
        # Save to pickle file
        filename = f'./testingdata/oig_instances_{size_category}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(grid_dataset, f)
        
        print(f"Saved {len(grid_dataset)} size categories to {filename}")
    
    # Generate a mixed random dataset for variety
    print("Generating random instances...")
    random_sizes = [20, 30, 40, 50]
    random_dataset = generator.generate_dataset('random', random_sizes)
    
    with open('./testingdata/oig_instances_random.pkl', 'wb') as f:
        pickle.dump(random_dataset, f)
    
    print("Instance generation completed!")

if __name__ == "__main__":
    main()