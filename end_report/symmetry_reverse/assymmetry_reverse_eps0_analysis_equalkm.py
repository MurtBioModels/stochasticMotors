from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.plotting import trajectories as traj
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import statistics as st
from motorgillespie.analysis import print_info as pi
from motorgillespie.analysis import test as test
import os
import pickle


dirct1 = '20221219_112505_symmetry_reverse_eps0_equalkm'
ts = [[1,1], [2,2], [3,3], [4,4]]
eps0ratiolist = [0.5, 0.67, 0.83, 1, 1.17, 1.33] # HAS TO BE THE RIGHT ORDER!!!!
# 0.3, 0.4, 0.5, 0.6, 0.7, 0.8



'''Dataframe'''
#df.rl_bead_n_parratio(dirct=dirct1, filename='', ts_list=ts, km_list=eps0ratiolist, parname='eps0')

'''Bead figures'''
bf.plot_n_parratio_rl(dirct=dirct1, filename='N_parratio_rl.csv', parname='eps0', figname='eps0', titlestring='', show=True)
