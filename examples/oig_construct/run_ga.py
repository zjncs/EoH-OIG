# run_ga.py
import random
import numpy as np
from tqdm import tqdm

from evaluation.instance import OIGInstance
from evaluation.solution import LeaderSolution, FollowerSolution
from evaluation.fitness import FitnessEvaluator

# --- 遗传算法参数 ---
POPULATION_SIZE = 50       # 种群大小 (p_max in paper)
MAX_GENERATIONS = 200      # 最大迭代次数 (n_maxIter)
TOURNAMENT_SIZE = 3        # 锦标赛选择大小 (K-way tournament)
MUTATION_RATE = 0.1        # 变异概率
CROSSOVER_RATE = 0.9       # 交叉概率

# 论文中提到的参数
REPAIR_NODES_TO_CHECK = 10 # 修复时检查的节点数
LEADER_BUDGET = 8          # 领导者拦截预算 (Ql)

def initialize_population(instance: OIGInstance) -> list[LeaderSolution]:
    """
    使用论文中的随机贪心算法 (Greedy) 初始化种群
    为简化，此处我们使用完全随机的方法初始化
    一个更完整的实现需要实现论文中的Greedy()
    """
    population = []
    for _ in range(POPULATION_SIZE):
        # 创建一个长度为节点数的0向量
        decision = np.zeros(instance.num_nodes, dtype=int)
        # 随机选择 Ql 个节点进行拦截
        interdict_indices = np.random.choice(instance.num_nodes, LEADER_BUDGET, replace=False)
        decision[interdict_indices] = 1
        population.append(LeaderSolution(decision))
    return population

def tournament_selection(population: list[LeaderSolution], fitness_evaluator: FitnessEvaluator) -> LeaderSolution:
    """K-锦标赛选择"""
    tournament = random.sample(population, TOURNAMENT_SIZE)
    # OIG是最小化问题，所以选择适应度最低的
    winner = min(tournament, key=lambda ind: fitness_evaluator.evaluate(ind))
    return winner

def crossover(parent1: LeaderSolution, parent2: LeaderSolution) -> LeaderSolution:
    """单点交叉"""
    decision1 = parent1.decision
    decision2 = parent2.decision
    size = len(decision1)
    
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, size - 1)
        child_decision = np.concatenate([decision1[:crossover_point], decision2[crossover_point:]])
        return LeaderSolution(child_decision)
    # 如果不交叉，随机返回一个父代
    return random.choice([parent1, parent2])

def mutate(individual: LeaderSolution) -> LeaderSolution:
    """位翻转变异 (0, 1, or 2 flips in paper)"""
    if random.random() < MUTATION_RATE:
        decision = individual.decision
        num_flips = random.choice([1, 2])
        for _ in range(num_flips):
            idx = random.randint(0, len(decision) - 1)
            decision[idx] = 1 - decision[idx] # 0 -> 1, 1 -> 0
    return individual

def repair(individual: LeaderSolution, instance: OIGInstance):
    """
    修复超出预算的个体
    论文策略：移除奖励最低的节点的拦截
    """
    while np.sum(individual.decision) > LEADER_BUDGET:
        # 找到所有被拦截的节点
        interdicted_indices = np.where(individual.decision == 1)[0]
        # 找到其中奖励最小的节点
        min_prize_idx = -1
        min_prize = float('inf')
        for idx in interdicted_indices:
            if instance.prizes[idx] < min_prize:
                min_prize = instance.prizes[idx]
                min_prize_idx = idx
        # 移除该节点的拦截
        if min_prize_idx != -1:
            individual.decision[min_prize_idx] = 0
        else: # 如果所有节点奖励一样，随机移除一个
            idx_to_remove = random.choice(interdicted_indices)
            individual.decision[idx_to_remove] = 0
            
    return individual

def main():
    """主函数"""
    # 1. 加载实例
    print("Loading instance...")
    instance = OIGInstance(f'C:/Users/zny/Desktop/eoh/EoH-main/examples/oig_construct/data/bayg29.tsp')
    instance.leader_budget = LEADER_BUDGET
    
    # 2. 初始化适应度评估器
    print("Initializing fitness evaluator...")
    # 论文中提到，初始解池C通过运行启发式算法k0次生成。
    # 这里为简化，我们先用一些随机的合法路线初始化
    initial_pool = [FollowerSolution.create_random_tour(instance) for _ in range(20)]
    fitness_evaluator = FitnessEvaluator(instance, initial_pool)

    # 3. 初始化种群
    print("Initializing population...")
    population = initialize_population(instance)

    # 4. 迭代进化
    print("Starting evolution...")
    best_solution_overall = None
    best_fitness_overall = float('inf')

    for gen in tqdm(range(MAX_GENERATIONS), desc="Generations"):
        new_population = []
        
        # 精英保留：保留上一代最好的个体
        best_of_last_gen = min(population, key=lambda ind: fitness_evaluator.evaluate(ind))
        new_population.append(best_of_last_gen)

        while len(new_population) < POPULATION_SIZE:
            # 选择
            parent1 = tournament_selection(population, fitness_evaluator)
            parent2 = tournament_selection(population, fitness_evaluator)
            
            # 交叉
            child = crossover(parent1, parent2)
            
            # 变异
            child = mutate(child)
            
            # 修复
            child = repair(child, instance)
            
            new_population.append(child)

        population = new_population

        # 更新全局最优解
        current_best_fitness = fitness_evaluator.evaluate(best_of_last_gen)
        if current_best_fitness < best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_solution_overall = best_of_last_gen
            print(f"\nNew best solution found in generation {gen+1}: Fitness = {best_fitness_overall:.2f}")

    # 5. 输出结果
    print("\nEvolution finished!")
    print(f"Best Leader Strategy Found: {np.where(best_solution_overall.decision == 1)[0]}")
    print(f"Estimated Best Follower Prize (minimized): {best_fitness_overall:.2f}")

if __name__ == '__main__':
    main()