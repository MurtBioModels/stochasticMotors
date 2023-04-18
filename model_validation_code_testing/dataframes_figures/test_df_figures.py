from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os
import motorgillespie.simulation.motor_class as mc

motor0_1 = mc.MotorFixed(k_t=0, f_ex=0)
motor0_1.runlength_cargo = [[1, 1, 1], [2], [3, 4]]
motor0_2 = mc.MotorFixed(k_t=0, f_ex=0)
motor0_2.runlength_cargo = [[2, 2, 2], [3], [4, 5]]
motor0_3 = mc.MotorFixed(k_t=0, f_ex=0)
motor0_3.runlength_cargo = [[3, 3, 3], [4], [5, 6]]

list = [motor0_1, motor0_2, motor0_3]
ts = '1'
km = ['0.2', '0.4', '0.6']
#
dict_rl = {}

for index, m in enumerate(list):
        #
        rl = m.runlength_cargo
        print(rl)
        rl_flat = [element for sublist in rl for element in sublist]
        print(rl_flat)
        #
        key = (ts, km[index])
        print(key)
        dict_rl[key] = rl_flat

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_rl.items() ]))
print(df)
df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'km']).dropna()
print(df_melt)


def TEST_xm_n_fex_km(dirct, filename, ts_list, fex_list, km_list, stepsize=0.01):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_xm = []
    key_tuples = []
    #
    teamsize_count = 0
    km_count = 0
    fex_count = 0
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
            print(f'km_count={km_count}')
            print(f'fex_count={fex_count}')
            # Unpickle test_motor0_1 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            fex = fex_list[fex_count]
            km = km_list[km_count]
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            ##### key tuple #####
            key_tuples.append(key)
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            xm_interpolated = []
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
                    print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    xm = motor.x_m_abs
                    del motor
                    print(f'len forces should be 1000: {len(xm)}')
                    #
                    print('Start interpolating distances...')
                    for i, value in enumerate(time):
                        # time points of run i
                        t = value
                        # locations of motors
                        xm_i = xm[i]
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
                        if len(t) != len(xm):
                            t.pop()
                        # Create function
                        f = interp1d(t, xm_i, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        mf_intrpl = f(t_intrpl)
                        # add nested list
                        xm_interpolated.extend(mf_intrpl)

            nested_xm.append(tuple(xm_interpolated))
            del xm_interpolated

            #
            if km_count < len(km_list) - 1:
                km_count += 1
            elif km_count == len(km_list) - 1:
                if fex_count == len(fex_list) - 1:
                    km_count = 0
                    fex_count = 0
                    teamsize_count += 1
                elif fex_count < len(fex_list) - 1:
                    km_count = 0
                    fex_count += 1
    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_xm)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples
    #
    mid_index = 15
    print(f'mid_index={mid_index}')
    print(f'multi_column[:mid_index]={multi_column[:mid_index]}')
    print(f'multi_column[mid_index:]={multi_column[mid_index:]}')
    #
    #
    df1 = pd.DataFrame(nested_xm[:mid_index], index=multi_column[:mid_index])
    print(df1)
    df2 = pd.DataFrame(nested_xm[mid_index:], index=multi_column[mid_index:])
    print(df2)
    del nested_xm
    df3 = pd.concat([df1, df2]).T
    del df1
    del df2
    #print(df3)
    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df3, value_name='xm', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_xm.csv')

    return

#
nested_xm = [(1,1,1,1,1), (2,2,2,2,2), (3,3,3,float('nan'),3), (4,4,float('nan'),float('nan'),float('nan'))]
key_tuples = [('[1]', '-1', '0.2'), ('[1]', '-1', '0.4'), ('[1]', '-2', '0.2'), ('[1]', '-2', '0.4')]
#
multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
print(multi_column)
del key_tuples
#
mid_index = 15
print(f'mid_index={mid_index}')
print(f'multi_column[:mid_index]={multi_column[:mid_index]}')
print(f'multi_column[mid_index:]={multi_column[mid_index:]}')
#
#
df1 = pd.DataFrame(nested_xm[:mid_index], index=multi_column[:mid_index])
print(df1)
df2 = pd.DataFrame(nested_xm[mid_index:], index=multi_column[mid_index:])
print(df2)
del nested_xm
df3 = pd.concat([df1, df2]).T
del df1
del df2
print(df3)
'''
#
print('Make dataframe from dictionary... ')
df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
print(df)
'''
print('Melt dataframe... ')
df_melt = pd.melt(df3, value_name='xm', var_name=['team_size', 'f_ex', 'km']).dropna()
print(df_melt)
