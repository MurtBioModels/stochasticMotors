import motorgillespie.simulation.variable_loops as vl
import time
import cProfile
import pstats
import numpy as np

### Constants ###
Boltzmann = 1.38064852e-23
### Motor parameters ###
kinesin_params = {
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
 'init_state': 'unbound',
 'calc_eps': 'exponential', # exponential and gaussian
}

dynesin_params = {
 'family': 'Kinesin-1',
 'member': 'retro',
 'step_size': 8,
 'k_m': None,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': 0.66,
 'f_d': 2.1,
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
    't_max': None,
    'dimension': '1D'
}

date = time.strftime("%Y%m%d_%H%M%S")
dir = f'{date}_teamsize_km_symbreak1_no_endtime'

team_comb = [[4,4]]
#retro_km = np.arange(0.02, 0.2, 0.02)
retro_km = [0.02, 0.1, 0.2]

for i in team_comb:
    for j in retro_km:
        gill_set['n_motors'] = i
        dynesin_params['k_m'] = j

        ### Data storage ###
        subdir = f'{i}n_{j}dynkm'
        short_description = ''

        # Initiate motor team an run simulation n_it times for t_end seconds each
        with cProfile.Profile() as profile:
            out = output_gillespie = vl.init_run(sim_params, gill_set, dynesin_params,  kinesin_params,  sd=short_description, dirct=dir, subdir=subdir)
            ps = pstats.Stats(profile)
            ps.sort_stats('calls', 'cumtime')
            ps.print_stats(15)

