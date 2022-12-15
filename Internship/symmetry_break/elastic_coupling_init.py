import numpy as np
import motorgillespie.simulation.variable_loops as vl
import time
import cProfile
import pstats

### Constants ###
Boltzmann = 1.38064852e-23
### Motor parameters ###
kinesin_params = {
 'family': 'Kinesin-1',
 'member': 'antero',
 'step_size': 8,
 'k_m': None,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': 0.66,
# 'init_pos': 0.1,  
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'bound',
 'calc_eps': 'exponential',
}
'''
kinesin_params2 = {
 'family': 'Kinesin-1',
 'member': 'antero',
 'step_size': 8,
 'k_m': None,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': 0.66,
 #'init_pos': 0.1,
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'unbound',
 'calc_eps': 'exponential',
}
'''
### Simulation parameters ###
sim_params = {
 'k_t': 0,
 'f_ex': None
}

### Simulation settings ###
gill_set = {
    'n_motors': None,
    'n_it': 1000,
    't_max': 100,
    'dimension': '1D'
}

date = time.strftime("%Y%m%d_%H%M%S")
dirct = f'{date}_elastic_coupling'

#retro_km = np.arange(0.02, 0.22, 0.02)
retro_km = [0.02, 0.1, 0.2]
team_comb = [[1],[2],[3],[4]]
f_ex = [-4, -5, -6, -7]



for i in team_comb:
    for j in f_ex:
        for k in retro_km:
            gill_set['n_motors'] = i
            sim_params['f_ex'] = j
            kinesin_params['k_m'] = k
            #kinesin_params2['k_m'] = k

            ### Data storage ###
            subdir = f'{i}n_{j}fex_{k}km'
            short_description = ''

            # Initiate motor team an run simulation n_it times for t_end seconds each
            with cProfile.Profile() as profile:
                out = output_gillespie = vl.init_run(sim_params, gill_set, kinesin_params, sd=short_description, dirct=dirct, subdir=subdir)
                ps = pstats.Stats(profile)
                ps.sort_stats('calls', 'cumtime')
                ps.print_stats(15)


