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

#dirct1 = '20221107_111214_elastic_coupling_startbound0.1_withtrap'
#dirct2 = '20221114_114529_elastic_coupling_firsttryTEST_kt'

#df.meanmax_df2(dirct=dirct2, filename='', ts_list=['[1]', '[2]', '[3]'], km_list=['0.02', '0.1', '0.2'], fex_list=['0'],
       #      stepsize=0.01)
#bf.plot_N_km_meanmaxdist(dirct1, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=True)
#bf.plot_N_km_meanmaxdist(dirct2, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=True)
#bf.plot_N_km_runlength(dirct1, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=False)
#bf.plot_N_km_runlength(dirct2, 'N_km_meanmaxdist.csv', figname='', titlestring='', show=False)



def bla (dirct, filename):

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')


def test_xb(dirct, ts_list, kmratio_list):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    key_tuples = []
    #
    teamsize_count = 0
    km_ratio_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            sub_path = os.path.join(path,subdir)
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_ratio_count={km_ratio_count}')
            # Unpickle test_motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            key_tuples.append(key)
            #
            xm_interpolated = [] # not nested
            # loop through motor files
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file == 'motor0':
                        continue
                    if file == 'parameters.txt':
                        continue
                    if file == 'figures':
                        continue
                    if file == 'data':
                        continue
                    print('PRINT NAME IN FILES')
                    print(os.path.join(sub_path,file))

                    # Unpickle motor
                    print('Open pickle file...')
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    print('Done')
                    motor = pickle.load(pickle_file_motor)
                    print('Close pickle file...')
                    pickle_file_motor.close()
                    print('Done')
                    xm = motor.x_m_abs
                    print(f'len forces should be 1000: {len(xm)}')

                    print(f'{motor.id}, direction={motor.direction}, km={motor.k_m}')
                    print(f'motor locations={motor.x_m_abs}')
            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
