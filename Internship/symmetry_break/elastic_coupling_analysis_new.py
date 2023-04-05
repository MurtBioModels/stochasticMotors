from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.plotting import trajectories as traj
from motorgillespie.plotting import indlude_in_report as ir
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

## Simulation settings ##
dirct1 = '20230111_123825_elasticcoupling_100_False_allbound'
tslist = [1, 2, 3, 4]
kmlist = [0.2] # HAS TO BE THE RIGHT ORDER!!!!
fexlist = [0, -1, -2, -3, -4, -5, -6, -7]



'''Cargo RL pdf, cdf and lineplot/barplot <>'''
'''Cargo displacement: pdf and cdf'''
'''Cargo trajectory examples > + motor trajectories ??'''
'''bound motors barplot'''
'''(unbinding events)'''

''''Bind time cargo >> combine with RL? >>> or velocity??'''
'''segments or x vs v segments >> eventueel RL opsplitsen en dan segmenten'''

'''Motor forces pdf'''
'''Motor displacement pdf'''
'''Motor RL pdf, cdf and lineplot/barplot <>'''
''''contourplots??'''




'''Dataframe'''
#df.rl_bead_n_fex_km(dirct=dirct4, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist)
#df.fu_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist)
#df.bound_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01)
#df.motorforces_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.1)
#df.xm_n_fex_km(dirct=dirct1, filename='', ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01)

'''Bead figures'''
#bf.plot_n_fex_km_rl(dirct=dirct4, filename='_N_fex_km_rl.csv', figname='', titlestring='', show=True)
#bf.plot_fex_N_km_boundmotors(dirct=dirct1, filename='N_fex_km_boundmotors.csv', figname='', titlestring='', show=True)
#bf.plot_n_fex_km_xb(dirct=dirct4, filename, figname='', titlestring='', stat='probability', show=True)
'''Motor figures'''
#mf.plot_fex_N_km_fu_motors(dirct=dirct1, filename='N_fex_km_fu_motors.csv', figname='', titlestring='', show=True)
#mf.plot_fex_N_km_forces_motors(dirct=dirct1, filename1='N_fex_km_motorforces_tmTS3.csv', filename2='N_fex_km_motorforces_TS4.csv', figname='', titlestring='', total=False, show=True)
#mf.plot_fex_N_km_xm(dirct=dirct1, filename='_N_fex_km_xm.csv', figname='', titlestring='', show=True)

'''print info'''
#pi.time_scale(dirct=dirct2)

'''Trajectory'''
#traj.traj_fex(dirct=dirct1, subdir='[4]n_-5fex_0.02km', figname='[4]n_-5fex_0.02km', titlestring='N=4, f_ex=-5pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[4]n_-5fex_0.2km', figname='[4]n_-5fex_0.2km', titlestring='N=4, f_ex=-5pN, km=0.2pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[4]n_-7fex_0.02km', figname='[4]n_-7fex_0.02km', titlestring='N=4, f_ex=-7pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[4]n_-7fex_0.2km', figname='[4]n_-7fex_0.2km', titlestring='N=4, f_ex=-7pN, km=0.2pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[2]n_-7fex_0.02km', figname='[2]n_-7fex_0.02km', titlestring='N=2, f_ex=-7pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[2]n_-7fex_0.2km', figname='[2]n_-7fex_0.2km', titlestring='N=2, f_ex=-7pN, km=0.2pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[4]n_-6fex_0.02km', figname='[4]n_-6fex_0.02km', titlestring='N=4, f_ex=-6pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct1, subdir='[4]n_-6fex_0.2km', figname='[4]n_-6fex_0.2km', titlestring='N=4, f_ex=-6pN, km=0.2pN/nm', it=0, show=True)



#traj.traj_fex(dirct=dirct2, subdir='[1]n_0fex_0.2km', figname='[1]n_0fex_0.2km', titlestring='N=1, f_ex=0 pN, km=0.2pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct2, subdir='[1]n_0fex_0.02km', figname='[1]n_0fex_0.02km', titlestring='N=1, f_ex=0 pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct2, subdir='[4]n_0fex_0.02km', figname='[4]n_0fex_0.02km', titlestring='N=4, f_ex=0pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct2, subdir='[4]n_0fex_0.2km', figname='[4]n_0fex_0.2km', titlestring='N=4, f_ex=0pN, km=0.2pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct2, subdir='[4]n_0fex_0.02km', figname='[4]n_0fex_0.02km', titlestring='N=4, f_ex=0pN, km=0.02pN/nm', it=0, show=True)
#traj.traj_fex(dirct=dirct2, subdir='[4]n_0fex_0.2km', figname='[4]n_0fex_0.2km', titlestring='N=4, f_ex=0pN, km=0.2pN/nm', it=0, show=True)
