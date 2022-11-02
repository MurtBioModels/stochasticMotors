from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct = '20221101_123744_elastic_coupling_withtrap'
file = 'runlength_N_km.csv'


bf.plot_N_km_runlength(dirct, figname='', titlestring='', file=file, show=False)
#bf.plot_N_km_bound_motors(dirct, figname='', titlestring='', show=False)

