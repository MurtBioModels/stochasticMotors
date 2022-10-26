from simulation2D import variable_loops as vl
import cProfile
import pstats
import numpy as np
import time


### Constants ###
Boltzmann = 1.38064852e-23

### Protein parameters ###
motor_params_antero = {
 'family': 'Kinesin-1',
 'member': 'unknown',
 'step_size': 8,
 'k_m': 0.21,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': [0.66],
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'anterograde',
 'init_state': 'bound',
 'test1': 1000
}
motor_params_retro = {
 'family': 'Kinesin-1',
 'member': 'unknown',
 'step_size': 8,
 'k_m': 0.21,
 'v_0': 740,
 'alfa_0': 92.5,
 'f_s': 7,
 'epsilon_0': [0.66],
 'f_d': 2.1,
 'bind_rate': 5,
 'direction': 'retrograde',
 'init_state': 'bound',
 'test1': 10000
}

### Simulation parameters ###
sim_params = {
 'dp_v1': np.array([2.90, 2.25]),
 'dp_v2': np.array([0, 0.18]),
 'radius': 200,
 'rest_length': 35,
 'temp': 4.1/Boltzmann,
 'k_t': 0.1,
} #varvalue = None # Trap stiffness (pN/nm)

### Simulation settings ###
gill_set = {
    'n_motors': (1,1),
    'n_it': 1000,
    't_max': 100,
    'dimension': '1D',
    'epsilon': 'gaussian'
}

### Variable lists to loop over ###
kt_list = [0.08]

### Data storage ###
subject = 'model_validation'
date = time.strftime("%Y%m%d_%H%M%S")
subdir = f'{date}_sym'
short_description = 'This script runs a simulation with two equal motors, only one walks anterograde and and retrograde. \n' \
                    'The parameters are biological (so unbinding force = 7pN), unbinding is present at a constant rate and a trap stifness is present.\n' \
                    'The bead should have a mean walked distance of 0 \n' \
                    ''


## ##
if __name__ == '__main__':

    with cProfile.Profile() as profile:

        out = output_gillespie = vl.kt_loop_gillespie(kt_list, sim_params, gill_set, motor_params_antero, motor_params_retro, sd=short_description, dirct=subject, subdir=subdir)

        ps = pstats.Stats(profile)
        ps.sort_stats('calls', 'cumtime')
        ps.print_stats(15)

