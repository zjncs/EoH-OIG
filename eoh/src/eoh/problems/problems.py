# 在 problems/problems.py 文件中找到这一段：

class Probs():
    def __init__(self, paras):
        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from .optimization.tsp_greedy import run
            self.prob = run.TSPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "bp_online":
            from .optimization.bp_online import run
            self.prob = run.BPONLINE()
            print("- Prob "+paras.problem+" loaded ")
        # 在这里添加以下内容：
        elif paras.problem == "oig":
            from .optimization.oig import run
            self.prob = run.OIGPROBLEM()
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")
            print("Available problems: tsp_construct, bp_online, oig")  # 更新可用问题列表
    
    def get_problem(self):
        return self.prob