from scipy import stats
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os


def bead_symmetry(dirct, subdir):

    # Unpickle motor0
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    # Loop through lists in nested list of bead locations
    dict_total = {}
    xb_total = []
    for index, list_xb in enumerate(motor0.x_bead):

        # Calculate how long the bead is positioned at every location
        durations = np.diff(motor0.time_points[index])
        list_xb.pop() #for symmetry1 script
        xb_total.append(list_xb)
        for i, value in enumerate(list_xb):
            if value in dict_total:
                dict_total[value] += durations[i]
            else:
                dict_total[value] = durations[i]

    print(sum((key*value/sum(dict_total.values())) for key, value in dict_total.items()))
    flat_list = [item for sublist in xb_total for item in sublist]
    print(sum(flat_list))

    return dict_total


def intrpl_bead_symmetry(dirct, subdir, printstring):

    # Unpickle
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    xb_ip = []

    # Loop through lists in nested list of bead locations
    for index, list_xb in enumerate(motor0.x_bead):

        # Original data
        x = motor0.time_points[index]
        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
        #print(len(x))
        y = list_xb
        #print(len(y))
        if len(x) != len(y):
            x.pop()
        # Create function
        f = interp1d(x, y, kind='previous')
        # New x values, 100 seconds every second
        x_new = np.arange(0,1.5,0.001)
        # Do interpolation on new data points
        y_new = f(x_new)
        # Add interpolated datapoints to list
        xb_ip.extend(y_new)
    # Mean bead displacement
    mean_walked = sum(xb_ip)/len(xb_ip)
    print(f'The mean distance walked by the bead = {mean_walked} for {printstring}')

    return mean_walked


def xbead_ks_qq(dirct, subdir, interval=(0, 95), stepsize=0.001, hypothesis='norm'):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')


    # Unpickle motor0
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    xb_ip = []

    # Loop through lists in nested list of bead locations
    for index, list_xb in enumerate(motor0.x_bead):
        print(f'index={index}')

        # Original data
        t = motor0.time_points[index]
        xb = list_xb
        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
        if len(t) != len(xb):
            t.pop()
        # Create function
        f = interp1d(t, xb, kind='previous')
        # New x values, 100 seconds every second
        t_intrpl = np.arange(interval[0], interval[1], stepsize)
        # Do interpolation on new data points
        xb_intrpl = f(t_intrpl)

        # Add interpolated data points to list
        xb_ip.extend(xb_intrpl)

    statistic = stats.kstest(xb_ip, hypothesis)
    print(statistic)

    return statistic


def fair_event_choice(dirct, subdir, alt='two-sided'):
    """
    For a Gillespie simulation of x runs (each with t_end), tests if choice between antero- and retrograde motor
    is a fair coin. For total events per direction and first event per Gillespie run.
    Uses binam_test of Scipy.

    Parameters
    ----------

    Returns
    -------

    """
    # Unpickle motor_0 object
    pickleMotor0 = open(f'.\motor_objects\\{dirct}\{subdir}\Motor0', 'rb')
    motor0 = pickle.load(pickleMotor0)
    pickleMotor0.close()

    # Unpickle list of motor objects
    pickle_file_team = open(f'.\motor_objects\\{dirct}\{subdir}\MotorTeam', 'rb')
    motor_team = pickle.load(pickle_file_team)
    pickle_file_team.close()

    dict_succeses = {}
    dict_pvalues = {}
    # Loop through all lists in the match_event list
    for index, match in enumerate(motor0.match_events):
        print(index)
        count_dict = {}
        # Count the occurrence of each motor in the sublist
        for motor in motor_team:
            count_dict[f'{motor.id}'] = match.count(f'{motor.id}') # AANPASSEN

            if f'{motor.id}' not in dict_succeses.keys():
                print('motor_id not in dict_succeses.keys')
                dict_succeses[f'{motor.id}'] = []
                dict_succeses[f'{motor.id}'].append(match.count(f'{motor.id}')/len(match))
            else:
                dict_succeses[f'{motor.id}'].append(match.count(f'{motor.id}')/len(match))

        # Change of event if it was a fair event choice function
        p = 1/len(motor_team)
        # total number of matches in sublist
        n = len(match)
        # for each motor, do binominal test
        for key, value in count_dict.items():
            p_value = stats.binom_test(value, n=n, p=p, alternative=alt)
            if key not in dict_pvalues.keys():
                print('motor_id not in dict_pvalues.keys')
                dict_pvalues[key] = []
                dict_pvalues[key].append(p_value)
            else:
                dict_pvalues[key].append(p_value)

    # Plotting
    print('start plotting')
    for key, value in dict_pvalues.items():
        plt.hist(value, bins=len(value))
        plt.title(f'{key}')
        plt.xlabel('P values')
        #plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\pvalues{key}.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'p values < 0,05={len([i for i in value if i <= 0.05])} for motor {key}')

    for key, value in dict_succeses.items():
        plt.hist(value, bins=len(value))
        plt.title(f'{key}')
        plt.xlabel('successes')
        #plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\succeses{key}.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        return dict_succeses, dict_pvalues


def fair_first_step(dirct, subdir, interval=(0, 95), stepsize=0.001, n_exp=1000, p=0.5, alt='two-sided'):
    """

    Parameters
    ----------

    Returns
    -------

    """
    xb = []
    antero_counts = []
    p_values = []
    for i in range(n_exp+1):

        # Unpickle motor0 object
        pickle_file_motor0 = open(f'..\motor_objects\\{dirct}\\{subdir}\motor0_{i}', 'rb')
        motor0 = pickle.load(pickle_file_motor0)
        pickle_file_motor0.close()

        # Unpickle list of motor objects
        pickle_file_team = open(f'..\motor_objects\\{dirct}\{subdir}\MotorTeam_{i}', 'rb')
        motor_team = pickle.load(pickle_file_team)
        pickle_file_team.close()
        # List per experiment
        xb_ip = []
        # Loop through lists in nested list of bead locations
        for index, list_xb in enumerate(motor0.x_bead):
            print(f'index={index}')

            # Original data
            t = motor0.time_points[index]
            xb = list_xb
            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
            if len(t) != len(xb):
                t.pop()
            # Create function
            f = interp1d(t, xb, kind='previous')
            # New x values, 100 seconds every second
            t_intrpl = np.arange(interval[0], interval[1], stepsize)
            # Do interpolation on new data points
            xb_intrpl = f(t_intrpl)

            # Add interpolated data points to list
            xb_ip.extend(xb_intrpl)

        xb.append(sum(xb_ip)/len(xb_ip))

        antero_count = 0
        retro_count = 0
        for list in motor0.match_events:
            if list[0] == 'anterograde':
                antero_count += 1
            elif list[0] == 'retrograde':
                retro_count += 1
            else:
                print('what?')
        antero_counts.append(antero_count)

        p_value = stats.binom_test(antero_count, antero_count+retro_count, p=p, alternative=alt)
        p_values.append(p_value)

    plt.hist(antero_counts, bins=n_exp)
    plt.xlabel('Antero count')
    plt.show()

    plt.hist(p_values, bins=n_exp)
    plt.xlabel('P values')
    plt.show()

    plt.hist(xb, bins=n_exp)
    plt.xlabel('netto displacement bead over 100 runs iof 10s')
    plt.show()

    print(f'xb={sum(xb)/len(xb)}')
    print(f'{len([i for i in p_values if i <= 0.05])}')

    return antero_counts



