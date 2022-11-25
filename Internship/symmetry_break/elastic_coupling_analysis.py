from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct1 = '20221114_133645_elastic_coupling_fex'
tslist = [1, 2, 3, 4]
kmlist = [0.02, 0.04, 0.06, 0.08, 0.12, 0.14, 0.16, 0.18, 0.2, 0.1]
fexlist = [-0.5, -1, -2, -3, 0]

# making dataframes
#df.rl_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist)
#df.fu_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist)
#df.bound_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01)
#df.motorforces_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.1)



# Plotting
#bf.plot_fex_N_km_runlength(dirct=dirct1, filename='N_fex_km_rl.csv', figname='se_', titlestring='', show=True)
#mf.plot_fex_N_km_fu_motors(dirct=dirct1, filename='N_fex_km_fu_motors.csv', figname='', titlestring='', show=True)
#bf.plot_fex_N_km_boundmotors(dirct=dirct1, filename='N_fex_km_boundmotors.csv', figname='', titlestring='', show=True)
mf.plot_fex_N_km_forces_motors(dirct=dirct1, filename='N_fex_km_motorforces.tsv', figname='', titlestring='', show=True)
