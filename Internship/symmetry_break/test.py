import numpy as np
from scipy.interpolate import interp1d
import numpy
from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import dataframes as df
from motorgillespie.analysis import print_info as pi
import os
import pickle
import pandas as pd

'''
t = [0, 0.01, 0.03, 0.04, 0.09, 0.12, 0.18, 0.19, 0.24]
y = [0, 0, 8, 16, 24, float('nan'), float('nan'), 32, 40]

# Create function
f = interp1d(t, y, kind='previous')
# New x values, 100 seconds every second
interval = (0, t[-1])
t_intrpl = np.arange(interval[0], interval[1], 0.001)
# Do interpolation on new data points
xb_intrpl = f(t_intrpl)
# add nested list
print(t_intrpl)
print(xb_intrpl)

for index, value in enumerate(xb_intrpl):
    if np.isnan(value):
        print(f'index {value}={index}, and in t list value:{t_intrpl[index]}')
'''

dirct1 = '20221107_111214_elastic_coupling_startbound0.1_withtrap'
dirct2 = '20221114_114529_elastic_coupling_firsttryTEST_kt'

#df.meanmax_df2(dirct=dirct2, filename='', ts_list=['[1]', '[2]', '[3]'], km_list=['0.02', '0.1', '0.2'], fex_list=['0'],
       #      stepsize=0.01)
bf.plot_N_km_meanmaxdist(dirct1, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=True)
#bf.plot_N_km_meanmaxdist(dirct2, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=True)
#bf.plot_N_km_runlength(dirct1, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=False)
#bf.plot_N_km_runlength(dirct2, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=False)



def bla (dirct, filename):

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')


