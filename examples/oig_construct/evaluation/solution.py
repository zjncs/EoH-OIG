# evaluation/solution.py
import numpy as np
import random
from evaluation.instance import OIGInstance

class LeaderSolution:
    """领导者的解：一个二进制向量"""
    def __init__(self, decision: np.ndarray):
        self.decision = decision # shape: (num_nodes,)

class FollowerSolution:
    """跟随者的解：一条路线（节点序列）"""
    def __init__(self, tour: list[int]):
        self.tour = tour # e.g., [0, 5, 2, 8, 0]

    @staticmethod
    def create_random_tour(instance: OIGInstance) -> 'FollowerSolution':
        """创建一个不超预算的随机合法路线用于初始化"""
        nodes = list(range(1, instance.num_nodes))
        random.shuffle(nodes)
        
        current_tour = [instance.depot]
        current_length = 0.0
        
        for node in nodes:
            last_node = current_tour[-1]
            # 检查加入新节点和返回depot是否超预算
            new_length = current_length + instance.edge_weights[last_node, node] + instance.edge_weights[node, instance.depot]
            
            if new_length <= instance.follower_budget:
                current_length += instance.edge_weights[last_node, node]
                current_tour.append(node)
        
        current_tour.append(instance.depot)
        return FollowerSolution(current_tour)