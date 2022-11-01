from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct = '20221101_123744_elastic_coupling'
file = 'runlength_N_km.csv'
'''
# Create lists with
list_fn = []
list_ts = []

for j in retro_km:
    list_fn.append(f'(1,1)_{j}')
    list_ts.append(f'kin,dyn=(1,1)_{j}k_m')


for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
    for index, subdir in enumerate(subdirs):
        if subdir == 'figures':
            continue


    break
'''

#bf.cdf_xbead(dirct='20221025_164036_teamsize_km_symbreak1_(1,1)', figname='', titlestring='', show=False)
#bf.violin_xb(dirct=dirct, figname='', titlestring='', stepsize=0.001, show=False)
#bf.violin_trace_vel(dirct, figname='', titlestring='', show=False)
#bf.violin_fu_rl(dirct, k_t=0.0000001, figname='', titlestring='', show=False)
bf.plot_N_km(dirct, figname='', titlestring='', df=file, show=True)


