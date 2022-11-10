from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os

def bound_motors_df(dirct, filename, stepsize=0.01):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df_teamsize = ['2', '3', '4']
    df_km = ['0.02', '0.04', '0.06', '0.08', '0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df = pd.DataFrame(columns=['runlength', 'boundmotors',  'teamsize', 'km'])
    #
    teamsize_count = 0
    km_count = 0
    #
    counter = 0
    for root, subdirs, files in os.walk(f'.\motor_objects\\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            if subdir.startswith('[1]'):
                print(f'skipped:{subdir}')
                continue
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_count={km_count}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            tz = df_teamsize[teamsize_count]
            km = df_km[km_count]
            print(f'tz={tz}')
            print(f'km={km}')
            #
            runlength = motor0.runlength_bead
            antero_bound = motor0.antero_motors
            list_mean_bound = []

            print('Start interpolating bound motors')
            for index, list_bm in enumerate(antero_bound):
                    print(f'index={index}')
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
                    print(f'mean_bound={mean_bound}')
                    list_mean_bound.append(mean_bound)

            for i, value in enumerate(runlength):
                df.loc[counter, 'runlength'] = value
                df.loc[counter, 'boundmotors'] = list_mean_bound[i]
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
    print(df)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_km_meanbound.csv')

    return

def meanmax_df(dirct, filename, stepsize=0.01):
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


def df_N_km_forcesmotors(dirct, stepsize=0.01):
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
    dict_forces = {}
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
            list_forces = []
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
                    if subdir.startswith('[4]'):
                        print(f'skipped:{subdir}')
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
                    forces = motor.forces[i]
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
                    if len(t) != len(forces):
                        t.pop()
                    # Create function
                    f = interp1d(t, forces, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    forces_interpl = f(t_intrpl)
                    # Remove zeroes
                    forces_intrpl_nozero = [x for x in forces_interpl if x != 0]
                    list_check = [x for x in forces_intrpl_nozero if x < 0]
                    if tz == '1':
                        if len(list_check) > 0:
                            AssertionError()
                    if 0 in forces_intrpl_nozero:
                        print(f'something went wrong with the zeroes')
                    # Add interpolated data points to list of all forces of one motor
                    list_forces.extend(forces_intrpl_nozero)

            #
            if tz in dict_forces.keys():
                dict_forces[tz][km] = list_forces
            else:
                dict_forces[tz] = {}
                dict_forces[tz][km] = list_forces
            print(f'keys dict forces: {dict_forces.keys()}')

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
    df = pd.DataFrame.from_records([(key, key2, force) for key, value in dict_forces.items() for key2, value2 in value.items() for force in value2],
    columns=['teamsize', 'km', 'force'])

    print(df.head())
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\N_km_forces_motors.csv')

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
                    # Add interpolated data points to list of all forces of one motor
                    list_xm.extend(xm_intrpl_nozero)

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


def df_N_km_fu_motors(dirct):
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
    dict_forces = {}
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
            list_fu = []
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
            for motor in motor_team:
                print(f'motor: {motor.id}')
                list_fu.extend(motor.forces_unbind)

            #
            print(f'len list fu : {len(list_fu)}')
            #
            if tz in dict_forces.keys():
                dict_forces[tz][km] = list_fu
            else:
                dict_forces[tz] = {}
                dict_forces[tz][km] = list_fu
            print(f'keys dict forces: {dict_forces.keys()}')

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
    df = pd.DataFrame.from_records([(key, key2, force) for key, value in dict_forces.items() for key2, value2 in value.items() for force in value2],
    columns=['teamsize', 'km', 'fu'])

    print(df.head())
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\N_km_fu_motors.csv')

    return

def df_N_km_rl_motors(dirct):
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
    dict_forces = {}
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
            list_rl = []
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
            for motor in motor_team:
                print(f'motor: {motor.id}')
                list_rl.extend(motor.run_length)

            #
            print(f'len list RL : {len(list_rl)}')
            #
            if tz in dict_forces.keys():
                dict_forces[tz][km] = list_rl
            else:
                dict_forces[tz] = {}
                dict_forces[tz][km] = list_rl
            print(f'keys dict forces: {dict_forces.keys()}')

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
    df = pd.DataFrame.from_records([(key, key2, force) for key, value in dict_forces.items() for key2, value2 in value.items() for force in value2],
    columns=['teamsize', 'km', 'rl'])

    print(df.head())
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\N_km_rl_motors.csv')

    return
