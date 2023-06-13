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
 'f_d': 1000000,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'unbound',
 'calc_eps': 'exponential',
}

### Simulation parameters ###
sim_params = {
 'k_t': None,
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

### Data storage ###
date = time.strftime("%Y%m%d_%H%M%S")
kt = sim_params['k_t']
n_motors = gill_set['n_motors']
singlerun = gill_set['single_run']
init_state = 'notbound'
dirct = f'{date}_trap_{singlerun}_{init_state}'

kt_list = [0.12, 0.14, 0.16, 0.18, 0.2]
#kt_list = [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
# Initiate motor motor_team an run simulation n_it times for t_end seconds each
for i in kt_list:
    sim_params['k_t'] = i

    ### Data storage ###
    subdir = f'{i}_kt'
    short_description = ''

    # Initiate motor motor_team an run simulation n_it times for t_end seconds each
    with cProfile.Profile() as profile:
        out = output_gillespie = vl.init_run(sim_params, gill_set, plus_params, sd=short_description, dirct=dirct, subdir=subdir)
        ps = pstats.Stats(profile)
        ps.sort_stats('calls', 'cumtime')
        ps.print_stats(15)
