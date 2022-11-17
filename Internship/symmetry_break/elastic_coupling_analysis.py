from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct = '20221111_151921_elastic_coupling_firsttry'
# motor data
file1 = 'N_km_fu_motors.csv'
file2 = 'N_km_rl_motors.csv'
file3 = 'N_km_xm.csv'
file4 = 'N_km_fm.csv'
# cargo data
file5 = 'N_km_meanbound.csv'
file6 = 'N_km_meanmaxdist.csv'
file7 = ''


ts_list = ['1', '2', '3', '4', '5']
km_list = ['0.02', '0.1', '0.2']
fex_list = ['-0.5', '-1', '-2', '-3', '0']

# making dataframes

