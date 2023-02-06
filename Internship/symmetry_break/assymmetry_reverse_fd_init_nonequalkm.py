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
 'f_d': 2,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'unbound',
 'calc_eps': 'exponential', # exponential and gaussian
}

minus_params = {
 'family': 'Kinesin-5',
 'member': 'retro',
 'step_size': 8,
 'k_m': None,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': 0.66,
 'f_d': 1,
 'bind_rate': 5,
 'direction': 'retrograde',
 'init_state': 'unbound',
 'calc_eps': 'exponential',
}


### Simulation parameters ###
sim_params = {
 'k_t': 0,
 'f_ex': 0
}

### Simulation settings ###
gill_set = {
    'n_motors': None,
    'n_it': 1000,
    't_max': 100,
    'dimension': '1D',
    'single_run': False
}

date = time.strftime("%Y%m%d_%H%M%S")
dir = f'{date}_symmetry_reverse_fd2vs1'

team_comb = [[1,1], [2,2], [3,3], [4,4]]
km = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

for i in team_comb:
    for j in km:
        gill_set['n_motors'] = i
        minus_params['k_m'] = j

        ### Data storage ###
        subdir = f'{i}n_{j}minuskm'
        short_description = ''

        # Initiate motor team an run simulation n_it times for t_end seconds each
        with cProfile.Profile() as profile:
            out = output_gillespie = vl.init_run(sim_params, gill_set, plus_params, minus_params, sd=short_description, dirct=dir, subdir=subdir)
            ps = pstats.Stats(profile)
            ps.sort_stats('calls', 'cumtime')
            ps.print_stats(15)

