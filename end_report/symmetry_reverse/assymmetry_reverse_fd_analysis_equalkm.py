from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.plotting import trajectories as traj
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import statistics as st
from motorgillespie.analysis import print_info as pi
from motorgillespie.analysis import test as test
import os
import pickle


dirct1 = '20221215_132635_symmetry_reverse_fd_equalkm'
ts = [[1,1], [2,2], [3,3], [4,4]]
fdratiolist = [0.75, 0.5, 1.25, 1, 1.5] # HAS TO BE THE RIGHT ORDER!!!!
# 1.5, 0.5, 2.5, 1, 1.5



'''Dataframe'''
#df.rl_bead_n_parratio(dirct=dirct1, filename='', ts_list=ts, km_list=fdratiolist, parname='fd')

'''Bead figures'''
bf.plot_n_parratio_rl(dirct=dirct1, filename='N_parratio_rl.csv', parname='fd', figname='fd', titlestring='', show=True)
