import motorgillespie.simulation.variable_loops as vl
import time
import cProfile
import pstats

### Constants ###
Boltzmann = 1.38064852e-23
### Motor parameters ###
kinesin_params = {
 'family': 'Kinesin-1',
 'member': 'unknown',
 'step_size': 8,
 'k_m': 0.2,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': [0.66],
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'unbound',
 'test1':1000
}

dynesin_params = {
 'family': 'Kinesin-1',
 'member': 'unknown',
 'step_size': 8,
 'k_m': 0.2,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': [0.66],
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'retrograde',
 'init_state': 'unbound',
 'test1':1000
}


### Simulation parameters ###
sim_params = {
 'dp_v1': None,
 'dp_v2': None,
 'temp': None,
 'radius': None,
 'rest_length': None,
 'k_t': 0.08,
}

team_comb = [(1,1), (2,2)]
eps = ['constant', 'exponential', 'gaussian']
date = time.strftime("%Y%m%d_%H%M%S")
dir = f'{date}_symmetry1.2'
for i in team_comb:
    for j in eps:
        ### Simulation settings ###
        gill_set = {
            'n_motors': i,
            'n_it': 100,
            't_max': 100,
            'dimension': '1D',
            'epsilon': j
        }
        ### Data storage ###
        subdir = f'{i}_{j}'
        short_description = 'This Simulation runs with a team of anterograde- and retrograde motors \n' \
                            'The antero (kinesin_params) and retrograde (dynesin_params) motors differ only in walking direction, otherwise all Kinesin-1 parameters from literature \n' \
                            'done for while probs_sum <= rand >> motor_objects/symmetry1 and while probs_sum < rand >> motor_objects/symmetry1.2.'

        # Initiate motor team an run simulation n_it times for t_end seconds each
        with cProfile.Profile() as profile:
            out = output_gillespie = vl.init_run(sim_params, gill_set, kinesin_params, dynesin_params, sd=short_description, dirct=dir, subdir=subdir)
            ps = pstats.Stats(profile)
            ps.sort_stats('calls', 'cumtime')
            ps.print_stats(15)

