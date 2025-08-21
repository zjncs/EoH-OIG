# evaluation/fitness.py
from evaluation.instance import OIGInstance
from evaluation.solution import LeaderSolution, FollowerSolution
from utils.tour_operations import calculate_tour_length, calculate_tour_prize, improve_tour_by_insertion

class FitnessEvaluator:
    """
    负责评估一个领导者策略的适应度
    [cite_start]这对应论文中的 EstimateObjective 过程 [cite: 265]
    """
    def __init__(self, instance: OIGInstance, initial_solution_pool: list[FollowerSolution]):
        self.instance = instance
        self.follower_solution_pool = initial_solution_pool # 对应论文中的 C
        self.cache = {} # 缓存评估过的解

    def evaluate(self, leader_solution: LeaderSolution) -> float:
        """
        评估一个领导者策略的好坏（最小化问题）
        """
        # 使用缓存避免重复计算
        decision_tuple = tuple(leader_solution.decision)
        if decision_tuple in self.cache:
            return self.cache[decision_tuple]

        best_follower_prize = 0.0

        # [cite_start]遍历解池中的每一个路线 [cite: 266]
        for follower_sol in self.follower_solution_pool:
            # [cite_start]1. 修复路线：移除被拦截的节点 [cite: 245]
            repaired_tour = [node for node in follower_sol.tour if leader_solution.decision[node] == 0 or node == self.instance.depot]
            # 确保depot在且首尾一致
            if self.instance.depot not in repaired_tour:
                repaired_tour.insert(0, self.instance.depot)
            if repaired_tour[-1] != self.instance.depot:
                repaired_tour.append(self.instance.depot)
            
            # 如果修复后路线不合法，跳过
            if len(repaired_tour) < 2:
                continue
            
            # [cite_start]2. 改进路线：使用插入等操作 [cite: 246]
            # 论文中还用了2-opt，这里为简化，只用插入
            improved_tour = improve_tour_by_insertion(repaired_tour, self.instance, leader_solution.decision)
            
            # 3. 计算改进后路线的总奖励
            current_prize = calculate_tour_prize(improved_tour, self.instance, leader_solution.decision)
            
            if current_prize > best_follower_prize:
                best_follower_prize = current_prize

        # 领导者的目标是最小化跟随者的最大收益
        fitness = best_follower_prize
        self.cache[decision_tuple] = fitness
        return fitness