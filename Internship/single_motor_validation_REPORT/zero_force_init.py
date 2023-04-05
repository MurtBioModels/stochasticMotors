import motorgillespie.simulation.variable_loops as vl
import time
import cProfile
import pstats
import numpy as np

### Motor parameters ###
plus_params = {
 'family': 'Kinesin-1',
 'member': 'antero',
 'step_size': 8,
 'k_m': 0.2,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': 0.66,
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'bound',
 'calc_eps': 'exponential', # exponential and gaussian
}


### Simulation parameters ###
sim_params = {
 'k_t': 0,
 'f_ex': 0
}

### Simulation settings ###
gill_set = {
    'n_motors': [1],
    'n_it': 10000,
    't_max': 10000,
    'dimension': '1D'
}

date = time.strftime("%Y%m%d_%H%M%S")
dir = f'{date}_zeroforce_tend10000'



### Data storage ###
subdir = f'zero_force'
short_description = ''

# Initiate motor team an run simulation n_it times for t_end seconds each
with cProfile.Profile() as profile:
    out = output_gillespie = vl.init_run(sim_params, gill_set, plus_params, sd=short_description, dirct=dir, subdir=subdir)
    ps = pstats.Stats(profile)
    ps.sort_stats('calls', 'cumtime')
    ps.print_stats(15)

