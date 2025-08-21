import numpy as np
import pickle
from typing import Dict, List, Tuple, Any

class GetData:
    def __init__(self) -> None:
        self.datasets = {}
    
    def read_dataset_from_file(self, filename: str) -> Dict[str, Dict]:
        """Load OIG dataset from pickle file."""
        with open(filename, 'rb') as file:
            dataset = pickle.load(file)
        
        # Transform dataset format to match evaluation expectations
        transformed_dataset = {}
        
        for size, instances in dataset.items():
            dataset_name = f"OIG_{size}"
            transformed_dataset[dataset_name] = {}
            
            for instance_idx, instance_data in enumerate(instances):
                instance_name = f"test_{instance_idx}"
                transformed_dataset[dataset_name][instance_name] = instance_data
        
        return transformed_dataset
    
    def calculate_upper_bound(self, prizes: List[float]) -> float:
        """
        Calculate upper bound on follower's objective (all prizes without interdiction).
        This represents the maximum possible prize the follower could collect.
        """
        return sum(prizes)
    
    def calculate_trivial_lower_bound(self, prizes: List[float], interdiction_budget: int) -> float:
        """
        Calculate trivial lower bound by removing highest prizes.
        This is what happens if leader interdicts the k highest-prize nodes.
        """
        sorted_prizes = sorted(prizes, reverse=True)
        # Remove depot prize (should be 0) and top k prizes
        remaining_prizes = sorted_prizes[interdiction_budget + 1:]  # +1 for depot
        return sum(remaining_prizes) if remaining_prizes else 0
    
    def estimate_bounds_dataset(self, instances: Dict[str, Any], 
                               interdiction_budget: int) -> Dict[str, float]:
        """
        Calculate performance bounds for a dataset.
        Returns dictionary with upper bounds (no interdiction scenario).
        """
        bounds = {}
        
        for dataset_name, dataset_instances in instances.items():
            upper_bounds = []
            lower_bounds = []
            
            for instance_name, instance in dataset_instances.items():
                prizes = instance['prizes']
                
                # Upper bound: total prizes available
                upper_bound = self.calculate_upper_bound(prizes)
                upper_bounds.append(upper_bound)
                
                # Lower bound: greedy interdiction of highest prizes
                lower_bound = self.calculate_trivial_lower_bound(prizes, interdiction_budget)
                lower_bounds.append(lower_bound)
            
            # Store average bounds for this dataset
            bounds[dataset_name] = {
                'upper_bound': np.mean(upper_bounds),
                'lower_bound': np.mean(lower_bounds),
                'instances': list(dataset_instances.keys())
            }
        
        return bounds
    
    def get_instances(self, interdiction_budget: int, size: str) -> Tuple[Dict, Dict]:
        """
        Get instances and bounds for given interdiction budget and size category.
        
        Args:
            interdiction_budget: Number of nodes leader can interdict
            size: Size category ('small', 'medium', 'large', 'random')
            
        Returns:
            Tuple of (instances_dict, bounds_dict)
        """
        # Load appropriate dataset
        filename = f'./testingdata/oig_instances_{size}.pkl'
        try:
            self.datasets = self.read_dataset_from_file(filename)
        except FileNotFoundError:
            print(f"Dataset file {filename} not found. Run generate_oig_instances.py first.")
            return {}, {}
        
        # Calculate performance bounds
        bounds = self.estimate_bounds_dataset(self.datasets, interdiction_budget)
        
        print(f"Loaded {len(self.datasets)} dataset categories with interdiction budget {interdiction_budget}")
        for name, bound_info in bounds.items():
            print(f"  {name}: Upper bound = {bound_info['upper_bound']:.2f}, "
                  f"Lower bound = {bound_info['lower_bound']:.2f}")
        
        return self.datasets, bounds
    
    def calculate_instance_statistics(self, instance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various statistics for a single instance."""
        nodes = instance['nodes']
        prizes = instance['prizes']
        
        # Basic statistics
        stats = {
            'n_nodes': len(nodes),
            'total_prize': sum(prizes),
            'avg_prize': np.mean(prizes[1:]),  # Exclude depot
            'max_prize': max(prizes),
            'distance_budget': instance['distance_budget']
        }
        
        # Graph density (avg distance between nodes)
        if len(nodes) > 1:
            distances = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    dx = nodes[i][0] - nodes[j][0]
                    dy = nodes[i][1] - nodes[j][1]
                    distances.append(np.sqrt(dx*dx + dy*dy))
            
            stats['avg_distance'] = np.mean(distances)
            stats['min_distance'] = np.min(distances)
            stats['max_distance'] = np.max(distances)
        
        return stats
    
    def analyze_dataset(self, size: str) -> None:
        """Analyze and print statistics about a dataset."""
        instances, bounds = self.get_instances(interdiction_budget=3, size=size)
        
        print(f"\n=== Dataset Analysis: {size} ===")
        
        for dataset_name, dataset_instances in instances.items():
            print(f"\nDataset: {dataset_name}")
            
            all_stats = []
            for instance_name, instance in dataset_instances.items():
                stats = self.calculate_instance_statistics(instance)
                all_stats.append(stats)
            
            # Aggregate statistics
            if all_stats:
                avg_stats = {
                    key: np.mean([s[key] for s in all_stats]) 
                    for key in all_stats[0].keys()
                }
                
                print(f"  Average nodes: {avg_stats['n_nodes']:.1f}")
                print(f"  Average total prize: {avg_stats['total_prize']:.1f}")
                print(f"  Average prize per node: {avg_stats['avg_prize']:.2f}")
                print(f"  Average distance budget: {avg_stats['distance_budget']:.2f}")

if __name__ == "__main__":
    # Test the data loading
    data_loader = GetData()
    
    # Analyze all datasets
    for size in ['small', 'medium', 'large', 'random']:
        try:
            data_loader.analyze_dataset(size)
        except Exception as e:
            print(f"Could not analyze {size} dataset: {e}")