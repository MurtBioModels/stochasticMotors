from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct1 = '20221201_151454_elastic_coupling_fex_allbound'
tslist = [1, 2, 3, 4]
kmlist = [0.02, 0.1, 0.2] # HAS TO BE THE RIGHT ORDER!!!!
#fexlist = [-4, -5, -6, -7]
fexlist = [-0.5, -1, -2, -3]


# making dataframes
#df.rl_bead_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist)
#df.fu_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist)
#df.bound_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01)
#df.motorforces_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.1)
#df.xm_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01)
# Plotting
#bf.plot_n_fex_km_rl(dirct=dirct1, filename='N_fex_km_rl.csv', figname='se_', titlestring='', show=True)
#mf.plot_fex_N_km_fu_motors(dirct=dirct1, filename='N_fex_km_fu_motors.csv', figname='', titlestring='', show=True)
#bf.plot_fex_N_km_boundmotors(dirct=dirct1, filename='N_fex_km_boundmotors.csv', figname='', titlestring='', show=True)
#mf.plot_fex_N_km_forces_motors(dirct=dirct1, filename1='N_fex_km_motorforces_tmTS3.csv', filename2='N_fex_km_motorforces_TS4.csv', figname='', titlestring='', total=False, show=True)
#mf.plot_fex_N_km_xm(dirct=dirct1, filename='_N_fex_km_xm.csv', figname='', titlestring='', show=True)
pi.time_scale(dirct=dirct1)
