from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os

### N + FEX + KM >> ELASTIC C. ###
def bound_motors_df(dirct, filename, ts_list, km_list, fex_list,  stepsize=0.01):
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
    df_teamsize = ts_list
    df_fex = fex_list
    df_km = km_list
    df = pd.DataFrame(columns=['runlength', 'mean_antero_bound', 'mean_retro_bound', 'teamsize', 'f_ex', 'km'])
    #
    teamsize_count = 0
    km_count = 0
    fex_count = 0
    #
    counter = 0
    for root, subdirs, files in os.walk(f'.\motor_objects\\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'fex_count={fex_count}')
            print(f'km_count={km_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = df_teamsize[teamsize_count]
            fex = df_fex[fex_count]
            km = df_km[km_count]
            print(f'tz={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            runlength = motor0.runlength_bead
            antero_bound = motor0.antero_motors
            retro_bound = motor0.retro_motors
            mean_antero_bound = []
            mean_retro_bound = []

            print('Start interpolating antero bound motors')
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
                    #print(f'mean_bound={mean_bound}')
                    mean_antero_bound.append(mean_bound)

            print('Start interpolating retro bound motors')
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

            for i, value in enumerate(runlength):
                df.loc[counter, 'runlength'] = value
                df.loc[counter, 'mean_antero_bound'] = mean_antero_bound[i]
                df.loc[counter, 'mean_retro_bound'] = mean_retro_bound[i]
                df.loc[counter, 'teamsize'] = ts
                df.loc[counter, 'f_ex'] = fex
                df.loc[counter, 'km'] = km
                #print(f'df.iloc[counter]={df.iloc[counter]}')
                counter +=1

            #
            if km_count < len(df_km) - 1:
                km_count += 1
            elif km_count == len(df_km) - 1:
                if fex_count == len(df_fex) - 1:
                    km_count = 0
                    fex_count = 0
                    teamsize_count += 1
                elif fex_count < len(df_fex) - 1:
                    km_count = 0
                    fex_count +=1

    print(df)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_km_meanbound.csv')

    return


def meanmaxdist_n_fex_km(dirct, filename, ts_list, km_list, fex_list, stepsize=0.01):
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
            # Unpickle test_motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            fex = km_list[fex_count]
            km = fex_list[km_count]
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
                    meanmax_distances.append('NaN')
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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_fex_km_meanmaxdist.csv')

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

            # Unpickle test_motor0 object
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


def df_N_km_xm(dirct, stepsize=0.01):
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
    df_teamsize = ['1','2', '3', '4']
    df_km = ['0.02', '0.04', '0.06', '0.08', '0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    #
    teamsize_count = 0
    km_count = 0
    #
    dict_xm = {}
    #
    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print('PRINT NAME IN SUBDIR')
            print(os.path.join(path,subdir))
            sub_path = os.path.join(path,subdir)
            # Unpickle test_motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            tz = str(df_teamsize[teamsize_count])
            km = str(df_km[km_count])
            print(f'tz={tz}')
            print(f'km={km}')
            #
            motor_team = []
            list_xm = []
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
            #
            length_motorteam = len(motor_team)
            print(f'teamsize:{length_motorteam}')

            #
            print('interpolation process')
            timepoints_len = len(motor0.time_points)
            print(f'timepointslen: {timepoints_len}')
            for motor in motor_team:
                print(f'motor: {motor.id}')
                for i in range(timepoints_len):
                    print(f'iteration {i} ')
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
                    xm_interpl = f(t_intrpl)
                    # Remove zeroes
                    xm_intrpl_nozero = [x for x in xm_interpl if x != 0]
                    list_check = [x for x in xm_intrpl_nozero if x < 0]
                    if tz == '1':
                        if len(list_check) > 0:
                            AssertionError()
                    if 0 in xm_intrpl_nozero:
                        print(f'something went wrong with the zeroes')
                    # Remove Nans
                    xm_intrpl_nonans = [x for x in xm_intrpl_nozero if np.isnan(x) == False]
                    # Add interpolated data points to list of all forces of one motor
                    list_xm.extend(xm_intrpl_nonans)

            #
            if tz in dict_xm.keys():
                dict_xm[tz][km] = list_xm
            else:
                dict_xm[tz] = {}
                dict_xm[tz][km] = list_xm
            print(f'keys dict forces: {dict_xm.keys()}')

            #
            if km_count < 9:
                km_count += 1
            elif km_count == 9:
                km_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    # make df here
    #df = pd.DataFrame.from_dict({(i,j): dict_forces[i][j]
                           #for i in dict_forces.keys()
                           #for j in dict_forces[i].keys()},
                           #)
    #df = df.melt(ignore_index=False, value_name="price").reset_index()
    #melted_xb = pd.melt(df, value_vars=df.columns, var_name='settings').dropna()
    df = pd.DataFrame.from_records([(key, key2, force) for key, value in dict_xm.items() for key2, value2 in value.items() for force in value2],
    columns=['teamsize', 'km', 'xm'])

    print(df.head())
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\N_km_xm.csv')

    return


### N + KM_RATIO >> SYM BREAK ###
def xb_N_kmratio_df(dirct, filename, ts_list, kmratio_list, stepsize=0.01):
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
    df_teamsize = ts_list
    df_km = kmratio_list
    df = pd.DataFrame(columns=['xb', 'teamsize', 'km_ratio'])
    #
    teamsize_count = 0
    km_ratio_count = 0

    #
    counter = 0
    for root, subdirs, files in os.walk(f'.\motor_objects\\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_ratio_count={km_ratio_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = df_teamsize[teamsize_count]
            km_ratio = df_km[km_ratio_count]
            print(f'tz={ts}')
            print(f'km_ratio={km_ratio}')
            #
            xb = motor0.x_bead
            list_xb = []

            print('Start interpolating xb')
            for index, list_xb in enumerate(xb):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    xb_i = list_xb
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(xb_i):
                        t.pop()
                    # Create function
                    f = interp1d(t, xb_i, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    xb_intrpl = f(t_intrpl)
                    # Remove NaNs
                    xb_intrpl_nonans = [x for x in xb_intrpl if np.isnan(x) == False]
                    #print(f'mean_bound={mean_bound}')
                    list_xb.extend(xb_intrpl_nonans)

            print('Append to list')
            for i, value in enumerate(list_xb):
                df.loc[counter, 'xb'] = value
                df.loc[counter, 'teamsize'] = ts
                df.loc[counter, 'km_ratio'] = km_ratio
                #print(f'df.iloc[counter]={df.iloc[counter]}')
                counter +=1


            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    print(df)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_xb.csv')

    return


def rl_n_kmr(dirct, filename, ts_list, kmratio_list):
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
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'tz={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            #
            runlength = list(motor0.runlength_bead)
            #
            dict_rl[key] = runlength
            #
            if km_ratio_count < len(kmratio_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmratio_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_rl.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_rl.csv')

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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_fu.csv')

    return


def meanmaxdist_n_kmr(dirct, filename, ts_list, kmratio_list, stepsize=0.01):
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
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
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
                    meanmax_distances.append('NaN')
            #
            print(f'len meanmaxdistances should be approx 1000: {len(meanmax_distances)}')

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
    dict_anterobound = {}
    dict_retrobound = {}
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
            ts = ts_list[teamsize_count]
            km_ratio = kmratio_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            key = (str(ts), str(km_ratio))
            print(f'key={key}')
            #
            #runlength = motor0.runlength_bead
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
                    #print(f'mean_bound={mean_bound}')
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

            dict_anterobound[key] = mean_antero_bound
            dict_retrobound[key] = mean_retro_bound
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
    df_antero = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_anterobound.items() ]))
    print(df_antero)
    df_antero_melt = pd.melt(df_antero, value_name='antero_bound', var_name=['team_size', 'km_ratio']).dropna()
    print(df_antero_melt)
    df_antero_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_anterobound.csv')
    #
    df_retro = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_retrobound.items() ]))
    print(df_retro)
    df_retro_melt = pd.melt(df_retro, value_name='retro_bound', var_name=['team_size', 'km_ratio']).dropna()
    print(df_retro_melt)
    df_retro_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_retrobound.csv')

    return
