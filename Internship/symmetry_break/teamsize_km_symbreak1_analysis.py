from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.plotting import trajectories as traj
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import statistics as st
from motorgillespie.analysis import print_info as pi
from motorgillespie.analysis import test as test
from motorgillespie.analysis import segment_trajectories as seg
import os
import pickle


dirct1 = '20221114_190539_teamsize_km_symbreak1'
tslist = [[1,1], [2,2], [3,3], [4,4]]
kmratiolist = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5, 1] # HAS TO BE THE RIGHT ORDER!!!!

'''Dataframe'''
#df.xb_N_kmratio_df(dirct=dirct1, filename='', ts_list=tslist, parratio_list=kmratiolist, stepsize=0.01)
#df.rl_bead_n_kmr(dirct=dirct1, filename='', ts_list=tslist, parratio_list=kmratiolist)
#df.fu_motors_n_kmr(dirct1, filename='', ts_list=tslist, parratio_list=kmratiolist)
#df.meanmaxdist_n_kmr(dirct=dirct1, filename='', ts_list=tslist, parratio_list=kmratiolist, stepsize=0.1)
#df.boundmotors_n_kmr(dirct=dirct1, filename='new', ts_list=tslist, parratio_list=kmratiolist, stepsize=0.01)
#df.motorforces_n_kmr(dirct=dirct1, filename='', ts_list=tslist, parratio_list=kmratiolist, stepsize=0.1)
#df.xb_n_kmr(dirct=dirct1, filename='', ts_list=tslist, parratio_list=kmratiolist, stepsize=0.01)
#df.xm_n_kmr(dirct=dirct1, filename='step0.01', ts_list=tslist, parratio_list=kmratiolist, stepsize=0.01)

'''Bead figures'''
#bf.plot_N_kmratio_boundmotors(dirct=dirct1, filename1='N_kmratio_anterobound.csv', filename2='N_kmratio_retrobound.csv', figname='', titlestring='antero=0.2pN/nm, retro=[0.02, 0.1, 0.2] antero first', show=True)
#bf.plot_N_kmr_xb(dirct=dirct1, filename='_N_kmratio_xb.csv', figname='', titlestring='', show=True)
#bf.distplots_rl(dirct=dirct1, filename='_2N_kmratio_rl.csv', figname='', titlestring='Retro=0.2pN/nm, antero=[0.02, 0.1, 0.2] (retro first)', show=True)
#mf.plot_N_km_motor_fu(dirct=dirct1, filename='N_kmratio_fu.csv', figname='', titlestring='retro=0.2pN/nm, antero=[0.02, 0.1, 0.2] antero'
                                                                                         #' first', show=True)
#bf.plot_n_kmratio_boundmotors(dirct=dirct1, filename='new_N_kmratio_anteroretrobound.csv', figname='', titlestring='', show=True)
#bf.plot_N_kmr_xb(dirct=dirct1, filename='_N_kmratio_xb.csv', figname='', titlestring='', show=False)
#bf.plot_n_kmratio_rl(dirct=dirct1, filename='N_kmratio_rl.csv', figname='', titlestring='', show=True)

'''Motor figures'''
#mf.plot_N_kmr_forces_motors(dirct=dirct1, filename='N_kmratio_motorforces.csv', figname='', titlestring='', show=True)
#mf.plot_N_kmr_forces_motors2(dirct=dirct1, filename='N_kmratio_motorforces.csv', figname='col_prob_cbtrue', titlestring='', show=False)
#mf.plot_N_kmr_xm(dirct=dirct1, filename='N_kmratio_xm.csv', figname='', titlestring='', show=False)
#mf.plot_Nkmr_mf_plusminus(dirct=dirct1, filename='N_kmratio_motorforces.csv', figname='withzeroes', titlestring='', show=True)


'''Statistics'''
#st.kstest_nkmratio_rl(dirct=dirct1, data_file='N_kmratio_rl.csv',filename_out='', team_size=tslist, km_ratio=kmratiolist)
#st.trying(dirct=dirct1, data_file='N_kmratio_rl.csv', team_size=tslist, km_ratio=kmratiolist)

'''Testing'''
#test.test_xb(dirct=dirct1, ts_list=tslist, parratio_list=kmratiolist)

'''Trajectory'''
#traj.traj_kmratio(dirct=dirct1, subdir='[4, 4]n_0.02dynkm', figname='[4, 4]n_0.02dynkm', titlestring='N=[4,4], plusmotor: 0.2pN/nm, minusmotor: 0.02pN/nm', it=0, show=True)
#traj.traj_kmratio(dirct=dirct1, subdir='[4, 4]n_0.2dynkm', figname='[4, 4]n_0.2dynkm', titlestring='N=[4,4], plusmotor: 0.2pN/nm, minusmotor: 0.2pN/nm', it=0, show=True)
#traj.traj_kmratio(dirct=dirct1, subdir='[1, 1]n_0.2dynkm', figname='[1, 1]n_0.2dynkm', titlestring='N=[1,1], plusmotor: 0.2pN/nm, minusmotor: 0.2pN/nm', it=0, show=True)
#traj.traj_kmratio(dirct=dirct1, subdir='[1, 1]n_0.02dynkm', figname='[1, 1]n_0.02dynkm', titlestring='N=[1,1], plusmotor: 0.2pN/nm, minusmotor: 0.02pN/nm', it=0, show=True)
#traj.traj_kmratio(dirct=dirct1, subdir='[4, 4]n_0.02dynkm', figname='[4, 4]n_0.02dynkm_zoom', titlestring='N=[4,4], plusmotor: 0.2pN/nm, minusmotor: 0.02pN/nm', it=0, show=True)


#seg.segment_parratio(dirct=dirct1)
