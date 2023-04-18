from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os
from motorgillespie.analysis import segment_trajectories as st

### N + FEX + KM >> ELASTIC C. ###
## cargo ##
def rl_bead_n_fex_km(dirct, filename, ts_list, fex_list, km_list):
    """

    Parameters
    ----------
    Check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    #
    dict_rl = {}
    #
    teamsize_count = 0
    km_count = 0
    fex_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'fex_count={fex_count}')
            print(f'km_count={km_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            fex = fex_list[fex_count]
            km = km_list[km_count]
            print(f'tz={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            #
            runlength = list(motor0.runlength_bead)
            #
            dict_rl[key] = runlength
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
                    fex_count +=1

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_rl.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_rl.csv')

    return

def fu_motors_n_fex_km(dirct, filename, ts_list, fex_list, km_list):
    """

    Parameters
    ----------
    check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_fu_motors = {}
    #
    teamsize_count = 0
    fex_count = 0
    km_count = 0
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
            #
            list_fu = []
            #
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
                    print('PRINT MOTOR FILE:')
                    print(os.path.join(sub_path,file))

                    # Unpickle motor
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    motor = pickle.load(pickle_file_motor)
                    pickle_file_motor.close()
                    #
                    print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    list_fu.extend(motor.forces_unbind)
            #
            dict_fu_motors[key] = list_fu
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
                    fex_count +=1

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_fu_motors.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='fu_motors', var_name=['team_size', 'f_ex', 'km_']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_fu_motors.csv')

    return

def bound_n_fex_km(dirct, filename, ts_list, fex_list, km_list, stepsize=0.01):
    """

    Parameters
    ----------
    ready
    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_boundmotors = {}
    #
    teamsize_count = 0
    km_count = 0
    fex_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'fex_count={fex_count}')
            print(f'km_count={km_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            fex = fex_list[fex_count]
            km = km_list[km_count]
            print(f'tz={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            #
            #runlength = motor0.runlength_bead
            motors_bound = motor0.antero_motors
            mean_bound_motors = []

            print('Start interpolating bound motors...')
            for index, list_bm in enumerate(motors_bound):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    bound = list_bm
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(bound):
                        t.pop()
                    # Create function
                    f = interp1d(t, bound, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    bound_intrpl = f(t_intrpl)
                    mean_bound = np.mean(bound_intrpl)
                    #print(f'mean_bound={mean_bound}')
                    mean_bound_motors.append(mean_bound)
            #
            dict_boundmotors[key] = mean_bound_motors
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
                    fex_count +=1

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    df_boundmotors = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_boundmotors.items() ]))
    print(df_boundmotors)
    df_bound_melt = pd.melt(df_boundmotors, value_name='bound_motors', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_bound_melt)
    df_bound_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_boundmotors.csv')

    return

def meanmaxdist_n_fex_km(dirct, filename, ts_list, fex_list, km_list, stepsize=0.01):
    """

    Parameters
    ----------
    check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_meanmaxdist = {}
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
            # Unpickle motor0 object
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
            #
            runlengths = motor0.runlength_bead
            meanmax_distances = [] # this will get 1000 entries
            #
            motor_team = []
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
                    print('PRINT MOTOR FILE:')
                    print(os.path.join(sub_path,file))

                    # Unpickle motor
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    motor = pickle.load(pickle_file_motor)
                    pickle_file_motor.close()
                    motor_team.append(motor)
            #
            print(f'team:{motor_team}')
            length_motorteam = len(motor_team)

            #
            print('Start interpolating distances...')
            for i, value in enumerate(runlengths):
                list_of_lists = [] # one run, so one nested list per motor
                for motor in motor_team:
                    # time points of run i
                    t = motor0.time_points[i]
                    # locations of motors
                    xm = motor.x_m_abs[i]
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(xm):
                        t.pop()
                    # Create function
                    f = interp1d(t, xm, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    xb_intrpl = f(t_intrpl)
                    # add nested list
                    list_of_lists.append(list(xb_intrpl))

                # check nested list
                print('BEGIN EDITING')
                test = [len(x) for x in list_of_lists]
                print(f'lists in listoflists should be of equal size: {test}')
                print(f'len(listoflists) should be {length_motorteam}: {len(list_of_lists)}')

                # zip nested list
                print('zip list...')
                zipped = list(zip(*list_of_lists))
                #print(f'print zipped: {zipped}')
                # check zipped list
                test2 = [len(x) for x in zipped]
                print(f'lists of zippedlists should be of equal size, namely {length_motorteam}: unqiue values= {np.unique(np.array(test2))}')
                print(f'len(zipped) should be same as {test}: {len(zipped)}')
                # remove nans
                print('Remove NaNs and lists that are smaller then 2...')
                nonans = [[y for y in x if np.isnan(y) == False] for x in zipped]
                nonans = [x for x in nonans if len(x) > 1]
                if len(nonans) > 0:
                    #print(f'print nozeroes: {nozeroes}')
                    # check if any zeroes
                    test3 = [x for sublist in nonans for x in sublist]
                    print(f'are there any NaNs? should not be: {test3.count(np.NaN)}')
                    # check equal sizes
                    test4 = [len(x) for x in nonans]
                    print(f'nozeroes lists should NOT be of equal size, unqiue values: {np.unique(np.array(test4))}')
                    # max distance
                    print('Sort lists...')
                    sortedlists = [sorted(x) for x in nonans]
                    # check sorted()
                    #print(f'before sort entry 0: {nozeroes[6]}')
                    #print(f'after sort entry 0: {sortedlists[6]}')
                    print('Calculate distance between leading and legging motor (max distance)...')
                    maxdistance = [x[-1]- x[0] for x in sortedlists]
                    #test if integer/floatL
                    print(f'check type first entry: {type(maxdistance[0])}')
                    # check len maxdistance
                    print('Calculate mean of the max distances...')
                    mean_maxdistance = sum(maxdistance)/len(maxdistance)
                    meanmax_distances.append(mean_maxdistance)
                else:
                    meanmax_distances.append(float("nan"))
            #
            print(f'len meanmaxdistances should be approx 1000: {len(meanmax_distances)}')

            #
            dict_meanmaxdist[key] = meanmax_distances
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
                    fex_count +=1

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_meanmaxdist.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='meanmaxdist_motors', var_name=['team_size', 'f_ex', 'km_ratio']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_meanmaxdist.csv')

    return

def meanmax_df_old(dirct, filename, stepsize=0.01):
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
    df_teamsize = ['2', '3', '4']
    df_km = ['0.02', '0.04', '0.06', '0.08', '0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df = pd.DataFrame(columns=['runlength', 'meanmaxdist', 'teamsize', 'km'])
    #
    teamsize_count = 0
    km_count = 0
    #
    counter = 0
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            if subdir.startswith('[1]'):
                print(f'skipped:{subdir}')
                continue
            print('PRINT NAME IN SUBDIR')
            print(os.path.join(path,subdir))
            sub_path = os.path.join(path,subdir)

            # Unpickle test_motor0_1 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            tz = df_teamsize[teamsize_count]
            km = df_km[km_count]
            print(f'tz={tz}')
            print(f'km={km}')
            #
            runlengths = motor0.runlength_bead
            meanmax_distances = [] # this will get 1000 entries
            motor_team = []
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
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    motor = pickle.load(pickle_file_motor)
                    pickle_file_motor.close()
                    motor_team.append(motor)

            print(f'team:{motor_team}')
            length_motorteam = len(motor_team)

            #
            for i, value in enumerate(runlengths):
                list_of_lists = [] # one run, so one nested list per motor
                for motor in motor_team:
                    # time points of run i
                    t = motor0.time_points[i]
                    # locations of motors
                    xm = motor.x_m_abs[i]
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(xm):
                        t.pop()
                    # Create function
                    f = interp1d(t, xm, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    xb_intrpl = f(t_intrpl)
                    # add nested list
                    list_of_lists.append(list(xb_intrpl))

                # check nested list
                print('BEGIN CHECKING')
                test = [len(x) for x in list_of_lists]
                print(f'lists in listoflists should be of equal size (BIG): {test}')
                print(f'lenlistoflists should be {length_motorteam}: {len(list_of_lists)}')

                # zip nested list
                zipped = list(zip(*list_of_lists))
                #print(f'print zipped: {zipped}')
                # check zipped list
                test2 = [len(x) for x in zipped]
                print(f'lists of zippedlists should be of equal size, namely {length_motorteam}: unqiue values= {np.unique(np.array(test2))}')
                print(f'lenzipped should be A LOT: {len(zipped)}')
                # remove zeroes
                nozeroes = [[y for y in x if y != 0] for x in zipped]
                nozeroes = [x for x in nozeroes if len(x) > 1]
                if len(nozeroes) > 1:
                    #print(f'print nozeroes: {nozeroes}')
                    # check if any zeroes
                    test3 = [x for sublist in nozeroes for x in sublist]
                    print(f'are there any zeroes? should not be: {test3.count(0)}')
                    # check equal sizes
                    test4 = [len(x) for x in nozeroes]
                    print(f'nozeroes lists should NOT be of equal size, unqiue values: {np.unique(np.array(test4))}')
                    # max distance
                    sortedlists = [sorted(x) for x in nozeroes]
                    # check sorted()
                    print(f'before sort entry 0: {nozeroes[0]}')
                    print(f'after sort entry 0: {sortedlists[0]}')

                    maxdistance = [x[-1]- x[0] for x in sortedlists]
                    #test if integer/floatL
                    print(f'check type first entry: {type(maxdistance[0])}')
                    # check len maxdistance
                    #print(f'len maxdistance should be equal to len zippedlist({len(zipped)}): {len(maxdistance)}')
                    mean_maxdistance = sum(maxdistance)/len(maxdistance)
                    meanmax_distances.append(mean_maxdistance)
                else:
                    meanmax_distances.append('NaN')

            print(f'len meanmaxdistances should be  1000: {len(meanmax_distances)}')
            for i, value in enumerate(runlengths):
                df.loc[counter, 'runlength'] = value
                df.loc[counter, 'meanmaxdist'] = meanmax_distances[i]
                df.loc[counter, 'teamsize'] = tz
                df.loc[counter, 'km'] = km
                print(f'df.iloc[counter]={df.iloc[counter]}')
                counter +=1

            #
            if km_count < 9:
                km_count += 1
            elif km_count == 9:
                km_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    print(f'df={df.head(10)}')
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_km_meanmaxdist.csv')

    return

## motors ##
def xm_n_fex_km(dirct, filename, ts_list, fex_list, km_list, stepsize=0.01):
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

            # Unpickle motor0 object
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
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
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
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km_ratio'])
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
    df_melt = pd.melt(df3, value_name='xm', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_xm.csv')

    return

def motorforces_n_fex_km(dirct, filename, ts_list, fex_list, km_list, stepsize=0.1):
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
    nested_motorforces = []
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
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
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
            key_tuples.append(key)
            #
            motor_interpolated_forces = [] # not nested
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
                    forces = motor.forces
                    print(f'len forces should be 1000: {len(forces)}')
                    del motor
                    #
                    #print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    #
                    print('Start interpolating forces...')
                    for i, value in enumerate(time):
                        #print(f'index={i}')
                        # time points of run i
                        t = value
                        #print(f't={t}')
                        # locations of motors
                        mf = forces[i]
                        if len(mf) < 2:
                            continue
                        #print(f'nf={mf}')
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                        if len(t) != len(mf):
                            t.pop()
                        # Create function
                        f = interp1d(t, mf, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        #print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        mf_intrpl = f(t_intrpl)
                        # add nested list
                        motor_interpolated_forces.extend(mf_intrpl)

            nested_motorforces.append(tuple(motor_interpolated_forces))
            del motor_interpolated_forces

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
                    fex_count +=1
    #
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_motorforces)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples
    #
    mid_index = 150
    print(f'mid_index={mid_index}')
    print(f'multi_column[:mid_index]={multi_column[:mid_index]}')
    print(f'multi_column[mid_index:]={multi_column[mid_index:]}')
    #
    df1 = pd.DataFrame(nested_motorforces[:mid_index], index=multi_column[:mid_index]).T
    print(df1)
    df2 = pd.DataFrame(nested_motorforces[mid_index:], index=multi_column[mid_index:]).T
    print(df2)
    del nested_motorforces
    #df3 = pd.concat([df1, df2]).T
    #del df1
    #del df2
    #print(df3)

    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframes_figures... ')
    df_melt1 = pd.melt(df1, value_name='motor_forces', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt1)
    del df1
    df_melt2 = pd.melt(df2, value_name='motor_forces', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt2)
    del df2
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    print('Save dataframe... ')
    df_melt1.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_fex_km_motorforces_tmTS3.csv')
    df_melt2.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_fex_km_motorforces_TS4.csv')

    return



### N + KM_RATIO >> SYM BREAK ###
def rl_cargo_n_kmr(dirct, filename, ts_list, kmminus_list):
    """
    DONE
    Parameters
    ----------


    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    #
    dict_rl = {}
    #
    teamsize_count = 0
    kmminus_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_minus_count={kmminus_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[kmminus_count]
            print(f'team_size={ts}')
            print(f'km_minus={km_minus}')
            #
            key = (str(ts), str(km_minus))
            print(f'key={key}')
            #
            runlength = list(motor0.runlength_bead)
            flat_rl = [element for sublist in runlength for element in sublist]
            #print(flat_rl)
            #
            dict_rl[key] = flat_rl
            #
            if kmminus_count < len(kmminus_list) - 1:
                kmminus_count += 1
            elif kmminus_count == len(kmminus_list) - 1:
                kmminus_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_rl.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_km_minus_rl.csv', index=False)

    return

def xb_n_kmr(dirct, filename, ts_list, kmratio_list, stepsize=0.01):
    """

    Parameters
    ----------
    check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_xb = {}
    #
    teamsize_count = 0
    km_ratio_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_ratio_count={km_ratio_count}')
            # Unpickle test_motor0_1 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            #
            xb = motor0.x_bead
            time = motor0.time_points
            del motor0
            xb_total_interpl = []
            print('Start interpolating bead locations...')
            for index, list_xb in enumerate(xb):
                    #print(f'index={index}')
                    # Original data
                    t_i = time[index]
                    xb_i = list_xb
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t_i) != len(xb_i):
                        t_i.pop()
                    # Create function
                    f = interp1d(t_i, xb_i, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t_i[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    xb_intrpl = f(t_intrpl)
                    xb_total_interpl.extend(xb_intrpl)

            #
            dict_xb[key] = xb_total_interpl

            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_xb.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='xb', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_kmratio_xb.csv')

    return

def fu_motors_n_kmr(dirct, filename, ts_list, kmratio_list):
    """

    Parameters
    ----------
    check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_fu_motors = {}
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
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            #
            list_fu = []
            #
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
                    print('PRINT MOTOR FILE:')
                    print(os.path.join(sub_path,file))

                    # Unpickle motor
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    motor = pickle.load(pickle_file_motor)
                    pickle_file_motor.close()
                    #
                    print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    list_fu.extend(motor.forces_unbind)
            #
            dict_fu_motors[key] = list_fu
            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_fu_motors.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='fu_motors', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_kmratio_fu.csv')

    return

def boundmotors_n_kmr(dirct, filename, ts_list, kmratio_list, stepsize=0.01):
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
    nested_anterobound = []
    nested_retrobound = []
    keys_anterobound = []
    keys_retrobound = []
    #
    teamsize_count = 0
    km_ratio_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_ratio_count={km_ratio_count}')
            # Unpickle test_motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key_antero = (str(ts), str(km_ratio), 'antero')
            key_retro = (str(ts), str(km_ratio), 'retro')
            print(f'key_antero={key_antero}')
            print(f'key_retro={key_retro}')
            keys_anterobound.append(key_antero)
            keys_retrobound.append(key_retro)
            #
            antero_bound = motor0.antero_motors
            retro_bound = motor0.retro_motors
            mean_antero_bound = []
            mean_retro_bound = []

            print('Start interpolating antero bound motors...')
            for index, list_bm in enumerate(antero_bound):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    bound = list_bm
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(bound):
                        t.pop()
                    # Create function
                    f = interp1d(t, bound, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    bound_intrpl = f(t_intrpl)
                    mean_bound = np.mean(bound_intrpl)
                    mean_antero_bound.append(mean_bound)


            print('Start interpolating retro bound motors...')
            for index, list_bm in enumerate(retro_bound):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    bound = list_bm
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(bound):
                        t.pop()
                    # Create function
                    f = interp1d(t, bound, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    bound_intrpl = f(t_intrpl)
                    mean_bound = np.mean(bound_intrpl)
                    #print(f'mean_bound={mean_bound}')
                    mean_retro_bound.append(mean_bound)
            #
            nested_anterobound.append(mean_antero_bound)
            nested_retrobound.append(mean_retro_bound)

            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    # ANTERO
    print(f'len(nested_nested_xb) should be {len(key_antero)}: {len(nested_anterobound)}')
    #
    multi_column_antero = pd.MultiIndex.from_tuples(key_antero, names=['team_size', 'km_ratio', 'direction'])
    print(multi_column_antero)
    del key_antero
    #
    df_antero = pd.DataFrame(nested_anterobound, index=multi_column_antero).T
    print(df_antero)
    del nested_anterobound
    #
    print('Melt antero dataframe... ')
    melt_antero = pd.melt(df_antero, value_name='motors_bound', var_name=['team_size', 'km_ratio', 'direction']).dropna()
    print(melt_antero)
    del df_antero

    # RETRO
    print(f'len(nested_nested_xb) should be {len(key_retro)}: {len(nested_retrobound)}')
    #
    multi_column_retro = pd.MultiIndex.from_tuples(key_retro, names=['team_size', 'km_ratio', 'direction'])
    print(multi_column_retro)
    del key_retro
    #
    df_retro = pd.DataFrame(nested_retrobound, index=multi_column_retro).T
    print(df_retro)
    del nested_retrobound
    #
    print('Melt retro dataframe... ')
    melt_retro = pd.melt(df_retro, value_name='motors_bound', var_name=['team_size', 'km_ratio', 'direction']).dropna()
    print(melt_retro)
    del df_retro
    # Merge two dataframes_figures
    print('Begin merging...')
    concat_df = pd.concat([melt_antero, melt_retro])
    print('Done')
    print(concat_df)
    concat_df.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_kmratio_anteroretrobound.csv')

    return

def meanmaxdist_n_kmr(dirct, filename, ts_list, kmratio_list, stepsize=0.01):
    """

    Parameters
    ----------
    NOT READY IN CALCULATIONS

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_meanmaxdist = {}
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
            # Unpickle test_motor0_1 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            #
            time = motor0.time_points
            meanmax_distances = [] # this will get 1000 entries
            #
            motor_team = []
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
                    print('PRINT MOTOR FILE:')
                    print(os.path.join(sub_path,file))

                    # Unpickle motor
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    motor = pickle.load(pickle_file_motor)
                    pickle_file_motor.close()
                    motor_team.append(motor)
            #
            print(f'team:{motor_team}')
            length_motorteam = len(motor_team)

            #
            print('Start interpolating distances...')
            for i, value in enumerate(time):
                list_of_lists = [] # one run, so one nested list per motor
                for motor in motor_team:
                    list_of_lists.append(motor.x_m_abs[i])

                # check nested list
                print('BEGIN EDITING')
                test = [len(x) for x in list_of_lists]
                print(f'lists in listoflists should be of equal size (1000): {test}')
                print(f'len(listoflists) should be {length_motorteam}: {len(list_of_lists)}')

                # zip nested list
                print('zip list...')
                zipped = list(zip(*list_of_lists))
                #print(f'print zipped: {zipped}')
                # check zipped list
                test2 = [len(x) for x in zipped]
                print(f'lists of zippedlists should be of equal size, namely {length_motorteam}: unqiue values= {np.unique(np.array(test2))} , type = {type(zipped[0])}')
                print(f'len(zipped) should be same as {test} (1000): {len(zipped)}')
                # remove nans
                print('Remove NaNs...')
                nonans = []
                for x in zipped:
                    nonans.append([y for y in x if y == y])
                #nonans = [list(filter(lambda x: x == x, inner_list)) for inner_list in zipped]
                if len(nonans) > 0:
                    #print(f'print nozeroes: {nozeroes}')
                    # check if any zeroes
                    test3 = [x for sublist in nonans for x in sublist if x != x]
                    print(f'are there any NaNs? should not be: {len(test3)}')
                    # check equal sizes
                    test4 = [len(x) for x in nonans]
                    print(f'nozeroes lists should NOT be of equal size, unqiue values: {np.unique(np.array(test4))}')
                    # max distance
                    print('Sort lists...')
                    sortedlists = [sorted(x) for x in nonans]
                    # check sorted()
                    #print(f'before sort entry 0: {nozeroes[6]}')
                    #print(f'after sort entry 0: {sortedlists[6]}')
                    print('Calculate distance between leading and legging motor (max distance)...')
                    maxdistance = [x[-1]- x[0] for x in sortedlists]
                    #test if integer/floatL
                    print(f'check type first entry: {type(maxdistance[0])}, and lenght: {len(maxdistance)}')
                    # check len maxdistance
                    print('Calculate mean of the max distances...')
                    t = np.diff(value)
                    print(f'len(diff(t)): {len(t)}')
                    mean_maxdistance = sum([a*b for a,b in zip(maxdistance,t)])/value[-1]
                    meanmax_distances.append(mean_maxdistance)
                else:
                    meanmax_distances.append(float('nan'))
            #
            print(f'len meanmaxdistances (approx 1000) : {len(meanmax_distances)}')

            #
            dict_meanmaxdist[key] = meanmax_distances
            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_meanmaxdist.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='meanmaxdist_motors', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_meanmaxdist.csv')

    return

def segment_parratio_asc(dirct, filename, ts_list, parratio_list):
    """

    Parameters
    ----------
    Check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_seg = {}
    #
    teamsize_count = 0
    parratio_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'parratio_count={parratio_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            par_ratio = parratio_list[parratio_count]
            print(f'ts={ts}')
            print(f'parratio={par_ratio}')
            #
            xb = motor0.x_bead
            t = motor0.time_points
            del motor0
            #
            key1 = (str(ts), str(par_ratio))
            print(f'key1={key1}')
            runs_asc, t_asc = st.diff_asc(x_list=xb, t_list=t)
            flat_runs_asc = [element for sublist in runs_asc for element in sublist]
            dict_seg[key1] = flat_runs_asc

            #key2 = (str(ts), str(par_ratio), 'vel')
            #print(f'key2={key2}')
            #flat_t_asc = [element for sublist in t_asc for element in sublist]
            #v_asc = [i/j for i, j in zip(flat_runs_asc, flat_t_asc)]
            #dict_seg[key2] = v_asc
            #
            if parratio_count < len(parratio_list) - 1:
                parratio_count += 1
            elif parratio_count == len(parratio_list) - 1:
                parratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_seg.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'par_ratio']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\asc_x_v{filename}.csv')

    return

def segment_parratio_desc(dirct, filename, ts_list, parratio_list, processive):
    """

    Parameters
    ----------
    Check

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_seg = {}
    #
    teamsize_count = 0
    parratio_count = 0
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            print('NEW SUBDIR/SIMULATION')
            print(os.path.join(path,subdir))
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'parratio_count={parratio_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            par_ratio = parratio_list[parratio_count]
            print(f'ts={ts}')
            print(f'parratio={par_ratio}')
            #
            xb = motor0.x_bead
            t = motor0.time_points
            del motor0
            #
            runs_desc, t_desc = st.diff_desc(x_list=xb, t_list=t)
            flat_runs_desc = [element for sublist in runs_desc for element in sublist]
            #
            key_x = (str(ts), str(par_ratio), 'x_seg')
            print(f'key={key_x}')
            dict_seg[key_x] = flat_runs_desc
            #
            key_v = (str(ts), str(par_ratio), 'v_seg')
            print(f'key={key_v}')
            flat_t_desc = [element for sublist in t_desc for element in sublist]
            v_desc =  [i/j for i, j in zip(flat_runs_desc, flat_t_asc)]
            #
            dict_seg[key_v] = v_desc
            #
            if parratio_count < len(parratio_list) - 1:
                parratio_count += 1
            elif parratio_count == len(parratio_list) - 1:
                parratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_seg.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\desc_x_v{filename}.csv')

    return


## motors ##
def xm_n_kmr(dirct, filename, ts_list, kmratio_list, stepsize=0.1):
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
            # Unpickle test_motor0_1 object
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
                    del motor
                    #
                    #print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    #
                    print('Start interpolating distances...')
                    for i, value in enumerate(time):
                        #print(f'index={i}')
                        # time points of run i
                        t = value
                        #print(f't={t}')
                        # locations of motors
                        xm_i = xm[i]
                        if len(xm_i) < 2:
                            continue
                        #print(f'nf={mf}')
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                        if len(t) != len(xm_i):
                            t.pop()
                        # Create function
                        f = interp1d(t, xm_i, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        #print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        xm_intrpl = f(t_intrpl)
                        # add nested list
                        xm_interpolated.extend(xm_intrpl)

            nested_xm.append(tuple(xm_interpolated))
            del xm_interpolated

            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_xm)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_ratio'])
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
    df_melt = pd.melt(df3, value_name='xm', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_xm.csv')

    return

def xm_n_kmr_2(dirct, filename, ts_list, kmratio_list, stepsize=0.1):
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
            # Unpickle motor_0 object
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
                    del motor
                    #
                    #print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    #
                    print('Start interpolating distances...')
                    for i, value in enumerate(time):
                        #print(f'index={i}')
                        # time points of run i
                        t = value
                        #print(f't={t}')
                        # locations of motors
                        xm_i = xm[i]
                        if len(xm_i) < 2:
                            continue
                        #print(f'nf={mf}')
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                        if len(t) != len(xm_i):
                            t.pop()
                        # Create function
                        f = interp1d(t, xm_i, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        #print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        xm_intrpl = f(t_intrpl)
                        # Random sampling
                        xm_sampled = random.sample(list(xm_intrpl), 50)
                        # add nested list
                        xm_interpolated.extend(xm_sampled)

            nested_xm.append(tuple(xm_interpolated))
            del xm_interpolated

            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_xm)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_ratio'])
    print(multi_column)
    del key_tuples
    #
    df = pd.DataFrame(nested_xm, index=multi_column)
    print(df)
    del nested_xm

    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='xm', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\N_kmratio_xm_{filename}.csv')

    return

def motorforces_n_kmr(dirct, filename, ts_list, parratio_list, stepsize=0.1):
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
    nested_motorforces = []
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
            # Unpickle motor_0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = parratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            key_tuples.append(key)
            #
            motor_interpolated_forces = [] # not nested
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
                    print(os.path.join(sub_path, file))

                    # Unpickle motor
                    print('Open pickle file...')
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    print('Done')
                    motor = pickle.load(pickle_file_motor)
                    print('Close pickle file...')
                    pickle_file_motor.close()
                    print('Done')
                    forces = motor.forces
                    print(f'len forces should be 1000: {len(forces)}')
                    del motor
                    #print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    #
                    print('Start interpolating distances...')
                    for i, value in enumerate(time):
                        #print(f'index={i}')
                        # time points of run i
                        t = value
                        #print(f't={t}')
                        # locations of motors
                        mf = forces[i]
                        if len(mf) < 2:
                            continue
                        #print(f'nf={mf}')
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                        if len(t) != len(mf):
                            t.pop()
                        # Create function
                        f = interp1d(t, mf, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        #print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        mf_intrpl = f(t_intrpl)
                        # add nested list
                        motor_interpolated_forces.extend(mf_intrpl)

            nested_motorforces.append(tuple(motor_interpolated_forces))
            del motor_interpolated_forces

            #
            if km_ratio_count < len(parratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(parratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_motorforces)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_ratio'])
    print(multi_column)
    del key_tuples
    #
    mid_index = 150
    print(f'mid_index={mid_index}')
    print(f'multi_column[:mid_index]={multi_column[:mid_index]}')
    print(f'multi_column[mid_index:]={multi_column[mid_index:]}')
    #
    df1 = pd.DataFrame(nested_motorforces[:mid_index], index=multi_column[:mid_index])
    print(df1)
    df2 = pd.DataFrame(nested_motorforces[mid_index:], index=multi_column[mid_index:])
    print(df2)
    del nested_motorforces
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
    df_melt = pd.melt(df3, value_name='motor_forces', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)

    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_motorforces.csv')

    return

def motorforces_n_kmr_2(dirct, filename, ts_list, parratio_list, stepsize=0.1):
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
    nested_motorforces = []
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
            # Unpickle motor_0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = parratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            key_tuples.append(key)
            #
            motor_interpolated_forces = [] # not nested
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
                    print(os.path.join(sub_path, file))

                    # Unpickle motor
                    print('Open pickle file...')
                    pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                    print('Done')
                    motor = pickle.load(pickle_file_motor)
                    print('Close pickle file...')
                    pickle_file_motor.close()
                    print('Done')
                    forces = motor.forces
                    print(f'len forces should be 1000: {len(forces)}')
                    del motor
                    #print(f'motor: {motor.id}, {motor.direction}, {motor.k_m}')
                    #
                    print('Start interpolating distances...')
                    for i, value in enumerate(time):
                        #print(f'index={i}')
                        # time points of run i
                        t = value
                        #print(f't={t}')
                        # locations of motors
                        mf = forces[i]
                        if len(mf) < 2:
                            continue
                        #print(f'nf={mf}')
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                        if len(t) != len(mf):
                            t.pop()
                        # Create function
                        f = interp1d(t, mf, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        #print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        mf_intrpl = f(t_intrpl)
                        # Random sampling
                        mf_sampled = random.sample(list(mf_intrpl), 50)
                        # add nested list
                        motor_interpolated_forces.extend(mf_sampled)

            nested_motorforces.append(tuple(motor_interpolated_forces))
            del motor_interpolated_forces

            #
            if km_ratio_count < len(parratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(parratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_motorforces)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_ratio'])
    print(multi_column)
    del key_tuples
    #
    df = pd.DataFrame(nested_motorforces, index=multi_column).T
    print(df)
    del nested_motorforces

    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='motor_forces', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)

    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_motorforces.csv')

    return




