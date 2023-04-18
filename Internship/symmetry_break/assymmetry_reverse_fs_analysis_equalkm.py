from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.plotting import trajectories as traj
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import statistics as st
from motorgillespie.analysis import print_info as pi
from motorgillespie.analysis import test as test
import os
import pickle


dirct1 = '20221214_143601_symmetry_reverse_fs_equalkm'
ts = [[1,1], [2,2], [3,3], [4,4]]
fsratiolist = [0.5, 0.57, 0.71, 0.88, 1] # HAS TO BE THE RIGHT ORDER!!!!




'''Dataframe'''
#df.rl_bead_n_parratio(dirct=dirct1, filename='', ts_list=ts, km_list=fsratiolist, parname='fs')

'''Bead figures'''
bf.plot_n_parratio_rl(dirct=dirct1, filename='N_parratio_rl.csv', parname='fs', figname='fs', titlestring='', show=False)
