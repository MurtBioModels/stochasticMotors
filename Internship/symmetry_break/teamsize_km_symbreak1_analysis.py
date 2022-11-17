from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle

dirct1 = '20221116_134017_teamsize_km_symbreak1_antero_km'
tslist = [[1,1], [2,2], [3,3], [4,4]]
#kmratiolist = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5, 1]
kmratiolist = [0.1, 0.5, 1]

#df.xb_N_kmratio_df(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=kmratiolist, stepsize=0.01)
#df.rl_N_kmratio_df(dirct=dirct1, filename='_2', ts_list=tslist, kmratio_list=kmratiolist)
#df.fu_motors_n_kmr(dirct1, filename='2_', ts_list=tslist, kmratio_list=kmratiolist)
df.meanmaxdist_n_kmr(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=kmratiolist, stepsize=0.01)

#bf.distplots_xb(dirct=dirct1, filename='N_kmratio_xb.csv', figname='', titlestring='', show=True)
#bf.distplots_rl(dirct=dirct1, filename='_2N_kmratio_rl.csv', figname='', titlestring='Retro=0.2pN/nm, antero=[0.02, 0.1, 0.2] (retro first)', show=True)
