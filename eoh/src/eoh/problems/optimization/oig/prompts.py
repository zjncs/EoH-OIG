# File: problems/optimization/oig/prompts.py

class GetPrompts():
    def __init__(self):
        self.prompt_task = """I need help designing a novel interdiction scoring function for the Orienteering Interdiction Game (OIG). 

In this bilevel optimization problem:
- A leader (interditor) can remove up to k nodes from a graph before the follower starts
- A follower then solves an orienteering problem to maximize collected prizes within a distance budget  
- The leader's goal is to minimize the follower's total prize by strategically choosing which nodes to interdict

Your task is to design a function that scores nodes for interdiction - higher scores indicate more valuable interdiction targets. The algorithm will select the top-k scored nodes to remove from the graph."""

        self.prompt_func_name = "interdiction_score"
        
        self.prompt_func_inputs = ['prizes', 'distances', 'depot', 'interdiction_budget']
        
        self.prompt_func_outputs = ['scores']
        
        self.prompt_inout_inf = """'prizes' is a list of prizes at each node (depot has prize 0).
'distances' is a 2D numpy array of distances between all pairs of nodes.
'depot' is the index of the depot node (usually 0) where the follower starts and ends.
'interdiction_budget' is the maximum number of nodes that can be interdicted.
The output 'scores' is a numpy array where scores[i] represents the value of interdicting node i (higher = more valuable)."""
        
        self.prompt_other_inf = """Important considerations:
- scores[depot] should be -inf (depot cannot be interdicted)
- Higher scores mean more valuable to interdict (will minimize follower's prize more)
- Consider both direct impact (removing high-prize nodes) and strategic impact (disrupting routing)
- Think about the follower's likely behavior: they prefer high prizes and short distances
- Consider graph connectivity: removing bridge nodes might force longer detours
- The function should be efficient and mathematically sound

Include 'import numpy as np' at the beginning of your function.
The novel function should be sufficiently sophisticated to outperform simple baselines like 'interdict highest prizes first'."""

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf
        
    def get_other_inf(self):
        return self.prompt_other_inf