# evaluation/instance.py
import tsplib95
import numpy as np

class OIGInstance:
    def __init__(self, filepath):
        problem = tsplib95.load(filepath)
        self.name = problem.name
        self.num_nodes = problem.dimension
        self.nodes = np.array(list(problem.node_coords.values()))
        self.edge_weights = self._calculate_distance_matrix()

        # [cite_start]根据论文生成奖励值 [cite: 311, 312]
        # p_i = 1 + (7141 * i + 73) mod 100
        self.prizes = np.array([1 + (7141 * (i + 1) + 73) % 100 for i in range(self.num_nodes)])
        
        # 默认值，可在主程序中设置
        self.depot = 0 #  depot node at index 0
        self.leader_budget = 5  # Ql
        
        # [cite_start]Bf = 0.5 * optimal TSP tour length [cite: 47, 313]
        # bayg29的最优TSP tour length是1610，这里硬编码用于示例
        self.follower_budget = 1610 * 0.5 

    def _calculate_distance_matrix(self):
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                dist = np.linalg.norm(self.nodes[i] - self.nodes[j])
                matrix[i, j] = matrix[j, i] = dist
        return matrix