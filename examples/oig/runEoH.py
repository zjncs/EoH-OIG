
from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initialization
paras = Paras() 

# Set parameters for OIG
paras.set_paras(method = "eoh",    
                problem = "oig",  # orienteering interdiction game
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = "your key",   # set your key
                llm_model = "deepseek-reasoner",
                ec_pop_size = 3,  # population size
                ec_n_pop = 2,     # number of generations  
                exp_n_proc = 2,   # multi-core parallel
                exp_debug_mode = False)

# Initialize evolution
evolution = eoh.EVOL(paras)

# Run evolution to discover interdiction heuristics
evolution.run()
