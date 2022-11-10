from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct = '20221107_152139_elastic_coupling_startbound0.1'
file1 = 'N_km_meanbound.csv'
file2 = 'N_km_meanmaxdist.csv'

file3 = 'N_km_fu_motors.csv'
file4 = 'N_km_rl_motors.csv'
file5 = 'N_km_xm.csv'
file6 = 'N_km_forces.csv'
#df.meanmax_df(dirct, file2, stepsize=0.01)
#df.bound_motors_df(dirct, file1, stepsize=0.01)

#bf.plot_N_km(dirct, figname='1', titlestring='', filename=file1, show=False)
#bf.plot_N_km(dirct, figname='2', titlestring='', filename=file2, show=False)

#pi.checkinggg(dirct)

bf.plot_N_km_meanmaxdist(dirct, filename=file2, figname=None, titlestring=None, show=False)
#bf.plot_N_km_boundmotors(dirct, filename=file1, figname=None, titlestring=None, show=False)

#df.df_N_km_forcesmotors(dirct, stepsize=0.1)
#mf.plot_N_km_motorforces(dirct, filename='N_km_forces.csv', figname=None, titlestring=None, show=False)

#df.df_N_km_xm(dirct, stepsize=0.1)
#mf.plot_N_km_xm(dirct, filename=file5, figname=None, titlestring=None, show=False)

#df.df_N_km_fu_motors(dirct)
#mf.plot_N_km_motor_fu(dirct, file3, figname=None, titlestring=None, show=False)

#df.df_N_km_rl_motors(dirct)
#mf.plot_N_km_motor_rl(dirct, filename=file4, figname=None, titlestring=None, show=False)
