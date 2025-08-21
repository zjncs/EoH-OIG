from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "bp_online", #['tsp_construct','bp_online']
                llm_api_endpoint = "https://generativelanguage.googleapis.com/v1/chat/completions", # set your LLM endpoint
                llm_api_key = "your key",   # set your key
                llm_model = "deepseek-chat",
                ec_pop_size = 2, # number of samples in each population
                ec_n_pop = 2,  # number of populations
                exp_n_proc = 2,  # multi-core parallel
                exp_debug_mode = False)

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()
