import motorgillespie.simulation.variable_loops as vl
import time
import cProfile
import pstats

### Motor parameters ###
plus_params = {
 'family': 'Kinesin-1',
 'member': 'plus',
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
 'calc_eps': 'exponential',
}

### Simulation parameters ###
sim_params = {
 'k_t': 0,
 'f_ex': 0
}

### Simulation settings ###
gill_set = {
    'n_motors': [1],
    'n_it': 1000,
    't_max': 100,
    'dimension': '1D',
    'single_run': False
}

fex = sim_params['f_ex']
n_motors = gill_set['n_motors']
singlerun = gill_set['single_run']
init_state = 'notbound'
date = time.strftime("%Y%m%d_%H%M%S")
dir = f'{date}_zeroforce_{singlerun}_{init_state}'
subdir = f'N={n_motors}_fex={fex}'


with cProfile.Profile() as profile:
    out = output_gillespie = vl.init_run(sim_params, gill_set, plus_params, dirct=dir, subdir=subdir)
    ps = pstats.Stats(profile)
    ps.sort_stats('calls', 'cumtime')
    ps.print_stats(15)

