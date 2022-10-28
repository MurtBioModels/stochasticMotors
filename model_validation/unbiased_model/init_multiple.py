from simulation2D import initiate_motors as im
from simulation2D import motor_class as mc
from simulation2D import gillespie_simulation as gs
import cProfile
import pstats
import numpy as np
import time
import os
import pickle

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
    'n_motors': None,
    'n_it': 100,
    't_max': 10,
    'dimension': '1D',
    '__epsilon': 'constant'
}
### Parameters to pass ###
n_motors = gill_set['n_motors']
kt = sim_params['k_t']
family = motor_params_antero['family']
n_exp = 100


### Data storage ###
subject = 'model_validation'
date = time.strftime("%Y%m%d_%H%M%S")
subdir = f'{date}_multiple'
short_description = 'This script runs a Gillespie simulation for n times. Each Simulation runs x times with each run' \
                    'having an end time t_max. Two identical motors are simulated, only one walks anterograde and and retrograde' \
                    'With the outputted motor objects, statistics on the events and data can be performed to validate if there is no bias' \
                    'present in the simulation and/or model.'
changes = ''

### Run Gillespie varval number of active motor proteins ###
if __name__ == '__main__':

    with cProfile.Profile() as profile:

        for i in range(n_exp+1):

            motor0 = im.init_motor_0(sim_params)
            motor_team = []
            motor_team.append(mc.MotorProtein(motor_params_antero['family'], motor_params_antero['member'], motor_params_antero['k_m'], motor_params_antero['alfa_0'], motor_params_antero['f_s'], motor_params_antero['epsilon_0'], motor_params_antero['f_d'], motor_params_antero['bind_rate'], motor_params_antero['step_size'], motor_params_antero['direction'], motor_params_antero['init_state'], 0))
            motor_team.append(mc.MotorProtein(motor_params_retro['family'], motor_params_retro['member'], motor_params_retro['k_m'], motor_params_retro['alfa_0'], motor_params_retro['f_s'], motor_params_retro['epsilon_0'], motor_params_retro['f_d'], motor_params_retro['bind_rate'], motor_params_retro['step_size'], motor_params_retro['direction'], motor_params_retro['init_state'], 1))

            team_out, motor0_out = gs.gillespie_2D_walk(motor_team, motor0, t_max=gill_set['t_max'], n_iteration=gill_set['n_it'], dimension=gill_set['dimension'], calc_epsilon=gill_set['__epsilon'])

            # Pickle motor objects
            if not os.path.isdir(f'..\motor_objects\\{subject}\\{subdir}'):
                os.makedirs(f'..\motor_objects\\{subject}\\{subdir}')

            pickleTeam = open(f'..\motor_objects\\{subject}\\{subdir}\motorteam_{i}', 'wb')
            pickle.dump(team_out, pickleTeam)
            pickleTeam.close()

            pickleMotor0 = open(f'..\motor_objects\\{subject}\\{subdir}\\motor0_{i}', 'wb')
            pickle.dump(motor0_out, pickleMotor0)
            pickleMotor0.close()

    with open(f"..\motor_objects\symmetry\{subdir}\parameters.txt", "w") as par_file:
        par_file.write(f"Simulation description: {short_description}: \n")
        par_file.write(f"Changes since previous run of the script: {changes}: \n")
        par_file.write("1st motor parameters: \n")
        for mp, value in motor_params_antero.items():
            par_file.write(f"{mp}={value}\n")
        par_file.write("2nd motor parameters: \n")
        for mp, value in motor_params_retro.items():
            par_file.write(f"{mp}={value}\n")
        par_file.write("Simulation parameters: \n")
        for sp, value in sim_params.items():
            par_file.write(f"{sp}={value}\n")
        par_file.write("Gillespie settings: \n")
        for gs, value in gill_set.items():
            par_file.write(f"{gs}={value}\n")


        ps = pstats.Stats(profile)
        ps.sort_stats('calls', 'cumtime')
        ps.print_stats(15)

