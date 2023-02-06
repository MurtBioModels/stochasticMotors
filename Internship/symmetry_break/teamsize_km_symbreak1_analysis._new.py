from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.plotting import trajectories as traj
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import statistics as st
from motorgillespie.analysis import print_info as pi
from motorgillespie.analysis import test as test
from motorgillespie.analysis import segment_trajectories as seg
from motorgillespie.plotting import indlude_in_report as ir
import os
import pickle


dirct1 = '20230115_211249_100_False_notbound'
tslist = [[1, 1], [2, 2], [3, 3], [4, 4]]
km_minus_list = [0.02, 0.04, 0.06, 0.08, 0.12, 0.14, 0.16, 0.18, 0.1, 0.2] # HAS TO BE THE RIGHT ORDER!!!!

'''Cargo RL pdf, cdf and lineplot/barplot <>'''
'''Cargo displacement: pdf and cdf'''
'''Cargo trajectory examples > + motor trajectories ??'''
'''Bound motors bar plot'''
'''(unbinding events)'''

'''Bind time cargo >> combine with RL? >>> or velocity??'''
'''segments or x vs v segments >> eventueel RL opsplitsen en dan segmenten'''

'''Motor forces pdf'''
'''Motor displacement pdf'''
'''Motor RL pdf, cdf and lineplot/barplot <>'''
'''Contour plots??'''

## Run length ##
#ir.rl_cargo_n_kmr(dirct=dirct1, filename='', ts_list=tslist, kmminus_list=km_minus_list)
#ir.plot_n_kmratio_rl(dirct=dirct1, filename='N_km_minus_rl.csv', figname='', titlestring='', show=True)

## Xb distribution ##
ir.xb_n_kmr(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=km_minus_list, stepsize=0.01)
