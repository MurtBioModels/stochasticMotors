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


dirct1 = '20230115_211249_symbreak_100_False_notbound'
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
#ir.plot_n_kmratio_rl(dirct=dirct1, filename='N_km_minus_rl.csv')

## Xb distribution ##
#ir.xb_n_kmr(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=km_minus_list)
#ir.plot_N_kmr_xb(dirct=dirct1, filename1='N_kmratio_xb1.csv', filename2='N_kmratio_xb2.csv', figname='everything', stat='probability', show=True)
#ir.xb_n_kmr_2(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=km_minus_list)
#ir.plot_N_kmr_xb_2(dirct=dirct1, filename='N_kmratio_xb.csv', figname='random_sampled', stat='probability', show=False)

## Bound Motors ##
#ir.boundmotors_n_kmr(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=km_minus_list, stepsize=0.1)

## Unbinding events ##
#ir.unbindevent_n_kmr(dirct=dirct1, filename='', ts_list=tslist, kmminus_list=km_minus_list)
#ir.plot_n_kmr_unbindevent(dirct=dirct1, filename='N_kmratio_unbindevents_.csv', figname='', titlestring='', n_include=('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]'), km_include=(0.1, 0.12, 0.14, 0.16, 0.18, 0.2), show=True)

## motor forces ##
#ir.motorforces_n_kmr(dirct=dirct1, filename='notsampled', ts_list=tslist, parratio_list=km_minus_list, stepsize=0.1)
#ir.motorforces_n_kmr_2(dirct=dirct1, filename='sampled', ts_list=tslist, parratio_list=km_minus_list, stepsize=0.1)
#ir.plot_N_kmr_forces_motors(dirct=dirct1, filename='N_kmratio_motorforces_notsampled.csv', figname='notsampled', titlestring='', n_include=('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]'), km_include=(0.1, 0.12, 0.14, 0.16, 0.18, 0.2), stat='probability', show=True)
#ir.plot_N_kmr_forces_motors(dirct=dirct1, filename='N_kmratio_motorforces_sampled.csv', figname='sampled', titlestring='', n_include=('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]'), km_include=(0.1, 0.12, 0.14, 0.16, 0.18, 0.2), stat='probability', show=True)
#ir.motorforces_n_kmr_2_sep(dirct=dirct1, filename='', ts_list=tslist, parratio_list=km_minus_list, stepsize=0.1)
#ir.plot_N_kmr_forces_motors_sep(dirct=dirct1, filename='N_kmratio_motorforces_sep.csv', figname='', titlestring='', n_include=('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]'), km_include=(0.1, 0.12, 0.14, 0.16, 0.18, 0.2), stat='probability', show=True)

## motor displacement ##
#ir.xm_n_kmr_2_sep(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=km_minus_list, stepsize=0.1)
ir.plot_N_kmr_xm_sep(dirct=dirct1, filename='N_kmratio_xm_sep_.csv', figname='stacktry', titlestring='', n_include=('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]'), km_include=(0.1, 0.12, 0.14, 0.16, 0.18, 0.2), stat='probability', show=True)
## motors run length ##
#ir.rl_motors_n_kmr_sep(dirct=dirct1, filename='', ts_list=tslist, kmratio_list=km_minus_list)
#ir.plot_n_kmratio_rl_motors_sep(dirct=dirct1, filename='N_kmratio_rl_sep_motors.csv', figname='', titlestring='', n_include=('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]'), km_include=(0.1, 0.12, 0.14, 0.16, 0.18, 0.2), show=True)
