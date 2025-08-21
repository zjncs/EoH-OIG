# utils/tour_operations.py
import numpy as np
# 在高层目录运行，需要调整导入路径
from evaluation.instance import OIGInstance

def calculate_tour_length(tour: list[int], instance: OIGInstance) -> float:
    """计算一条路线的总长度"""
    length = 0.0
    for i in range(len(tour) - 1):
        length += instance.edge_weights[tour[i], tour[i+1]]
    return length

def calculate_tour_prize(tour: list[int], instance: OIGInstance, leader_decision: np.ndarray) -> float:
    """计算一条路线在给定拦截策略下的总奖励"""
    prize = 0.0
    visited_nodes = set(tour)
    for node_idx in visited_nodes:
        if leader_decision[node_idx] == 0: # 如果未被拦截
            prize += instance.prizes[node_idx]
    return prize

def improve_tour_by_insertion(tour: list[int], instance: OIGInstance, leader_decision: np.ndarray) -> list[int]:
    """
    尝试将未访问的节点插入现有路线以增加奖励
    [cite_start]对应论文中的 Insert 操作 [cite: 246]
    """
    current_tour = list(tour)
    
    unvisited_nodes = [i for i in range(instance.num_nodes) if i not in current_tour and leader_decision[i] == 0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_insertion = None # (node_to_insert, position)

        for node_to_insert in unvisited_nodes:
            for i in range(len(current_tour) - 1):
                u, v = current_tour[i], current_tour[i+1]
                
                # 计算插入成本和收益
                cost_change = instance.edge_weights[u, node_to_insert] + instance.edge_weights[node_to_insert, v] - instance.edge_weights[u, v]
                new_length = calculate_tour_length(current_tour, instance) + cost_change
                
                if new_length <= instance.follower_budget:
                    # 收益是新节点的奖励
                    gain = instance.prizes[node_to_insert]
                    if gain > best_gain:
                        best_gain = gain
                        best_insertion = (node_to_insert, i + 1)

        if best_insertion:
            node, pos = best_insertion
            current_tour.insert(pos, node)
            unvisited_nodes.remove(node)
            improved = True
            
    return current_tour