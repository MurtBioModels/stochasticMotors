from scipy.interpolate import interp1d
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def xbead_dist(dirct, subdir, figname, titlestring, stepsize=0.001, stat='probability', show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """

    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')

    # Unpickle test_motor0
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
        interval = (0, t[-1])
        t_intrpl = np.arange(interval[0], interval[1], stepsize)
        # Do interpolation on new data points
        xb_intrpl = f(t_intrpl)
        # Remove zeroes
        xb_intrpl_nozeroes = [x for x in xb_intrpl if x != 0]
        # Remove Nans
        xb_intrpl_nonans = [x for x in xb_intrpl_nozeroes if np.isnan(x) == False]

        # Add interpolated data points to list
        xb_ip.extend(xb_intrpl_nonans)

    ### Plotting ###
    print('Making figure..')
    plt.figure()
    sns.displot(xb_ip, stat=stat)
    plt.title(f'Distribution (interpolated) of bead location {titlestring} ')
    plt.xlabel('Location [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\dist_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def xbead_dist_notintrpl(subdir, family, n_motors, kt, calc_eps, subject='varvalue', stat='probability'):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\{subject}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{subject}\{subdir}\\figures')
    #

    print(f'varvalue={kt}')
    # Unpickle motor0_Kinesin-1_0.0kt_1motors object
    pickle_file_motor0 = open(f'.\motor_objects\\{subject}\{subdir}\motor0_{family}_{kt}kt_{n_motors}motors', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    xb_nip = []

    # Loop through lists in nested list of bead locations
    for index, list_xb in enumerate(motor0.x_bead):
        print(f'index={index}')

        # Add interpolated data points to list
        xb_nip.extend(list_xb)

    # Plotting
    print('Making figure..')
    plt.figure()
    sns.displot(xb_nip, stat=stat, common_norm=False)
    plt.title(f'Distribution (NOT interpolated) of bead location varvalue={kt}')
    plt.xlabel('Location [nm]')
    plt.savefig(f'.\motor_objects\{subject}\{subdir}\\figures\\NOTintrpl_dist_xb_{calc_eps}eps_{kt}varvalue.png')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    return


def dist_act_motors1(dirct, subdir, figname, titlestring, stepsize=0.001, stat='probability', show=True):
    """
    Probability distribution of bound motors: interpolated

    Parameters
    ----------

    Returns
    -------

    """
    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')

    # Unpickle test_motor0
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    antero_bound = []
    retro_bound = []

    # Loop through lists in nested list of bead locations
    for index, antero_list in enumerate(motor0.antero_motors):
        print(f'index={index}')

        # Original data
        t = motor0.time_points[index]
        antero = antero_list
        retro = motor0.retro_motors[index]
        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
        if len(t) != len(antero):
            t.pop()
        # Create function
        f_antero = interp1d(t, antero, kind='previous')
        f_retro = interp1d(t, retro, kind='previous')
        # New x values, 100 seconds every second
        interval = (0, t[-1])
        t_intrpl = np.arange(interval[0], interval[1], stepsize)
        # Do interpolation on new data points
        antero_intrpl = f_antero(t_intrpl)
        retro_intrpl = f_retro(t_intrpl)
        # Add interpolated data points to list
        antero_bound.extend(antero_intrpl)
        retro_bound.extend(retro_intrpl)

    # Plotting
    print('Making figures..')
    plt.figure()
    plt.hist(antero_bound, density=True, label='antero')
    plt.hist(retro_bound, density=True, label='retro')
    plt.title(f'Bound antero- and retrograde motor distribution (interpolated): {titlestring}')
    plt.xlabel('Bound motors [n]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\\{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def dist_act_motors2(dirct, subdir, figname, titlestring, stat='probability', show=True):
    """
    Count of bound motors

    Parameters
    ----------

    Returns
    -------

    """
    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')

    # Unpickle test_motor0
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    #
    antero_bound = motor0.antero_motors
    retro_bound = motor0.retro_motors
    antero_flat = [val for sublist in antero_bound for val in sublist]
    retro_flat = [val for sublist in retro_bound for val in sublist]

    # Plotting
    print('Making figures..')
    plt.figure()
    sns.displot(antero_flat, stat=stat, common_norm=False)
    sns.displot(retro_flat, stat=stat, common_norm=False)
    plt.legend('anterograde', 'retrogarde')
    plt.title(f'Bound antero- and retrograde motor count: {titlestring}')
    plt.xlabel('Bound motors [n]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\\{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def stalltime_dist(dirct, subdir, figname, titlestring, stat='probability', show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')
    #
    pickle_file_motor0 = open(f'.\motor_objects\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    stalltime = [i for i in motor0.stall_time if i >= 0.25]

    # Plotting
    print('Making figure..')
    plt.figure()
    sns.displot(stalltime, stat=stat, common_norm=False)
    plt.title(f'Stall time distribution (>= 250 ms): {titlestring}')
    plt.xlabel('Stall time [s]')
    plt.savefig(f'.\motor_objects\\{dirct}\\{subdir}\\figures\\{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def rl_fu_bead(dirct, subdir, figname, titlestring, k_t, stat='count', show=True):
    """


    Parameters
    ----------

    Returns
    -------

    """

    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')

    # Unpickle motor_0 object
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    #
    rl_bead = motor0.runlength_bead
    fu_bead = [i*k_t for i in rl_bead]
    # Bins
    q25, q75 = np.percentile(rl_bead, [25, 75])
    bin_width = 2 * (q75 - q25) * len(rl_bead) ** (-1/3)
    bins_rl = round((max(rl_bead) - min(rl_bead)) / bin_width)
    q25, q75 = np.percentile(fu_bead, [25, 75])
    bin_width = 2 * (q75 - q25) * len(fu_bead) ** (-1/3)
    bins_fu = round((max(fu_bead) - min(fu_bead)) / bin_width)
    # Plotting
    print('Making figures..')
    plt.figure()
    sns.displot(rl_bead, stat=stat, bins=bins_rl)
    plt.title(f'Run length of bead: {titlestring}')
    plt.xlabel('Run length [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\\{subdir}\\figures\\dist_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()

    plt.figure()
    sns.displot(fu_bead, stat=stat, bins=bins_fu)
    plt.title(f'Unbinding force of bead: {titlestring}')
    plt.xlabel('Unbinding force [pN]')
    plt.savefig(f'.\motor_objects\{dirct}\\{subdir}\\figures\\dist_fu_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def trace_velocity(dirct, subdir, figname, titlestring,  stat='count', show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\{subdir}\\figures')

    #
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()
    #
    trace_velocities = []
    # Loop through lists in nested list
    for index, xb_list in enumerate(motor0.x_bead):
        xb = xb_list
        t = motor0.time_points[index]
        vel = (xb[-1]-xb[0])/(t[-1]-t[0])
        trace_velocities.append(vel)

    # Bins
    q25, q75 = np.percentile(trace_velocities, [25, 75])
    bin_width = 2 * (q75 - q25) * len(trace_velocities) ** (-1/3)
    bins = round((max(trace_velocities) - min(trace_velocities)) / bin_width)
    # Plot distributions
    plt.figure()
    sns.displot(trace_velocities, stat=stat, bins=bins, common_norm=False, palette="bright")
    plt.title(f'Trace velocity distribution: {titlestring}')
    plt.xlabel('Trace velocity [nm/s]')
    plt.savefig(f'.\motor_objects\\{dirct}\\{subdir}\\figures\\dist_trace_vel_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def segmented_velocity(dirct, subdir, figname, titlestring, stat='probability', show=True):
    """
    velocities between two events
    Parameters
    ----------

    Returns
    -------

    """
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\{subdir}\\figures')

    # Unpickle test_motor0
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    vel_inst = []

    # Loop through lists in nested list of bead locations
    for index, list_xb in enumerate(motor0.x_bead):
        print(f'index={index}')
        t = motor0.time_points[index]
        if len(t) != len(list_xb):
            t.pop()
        xb_diff = np.diff(list_xb)
        t_diff = np.diff(t)
        v =  [ i / j for i, j in zip(xb_diff, t_diff) ]
        vel_inst.extend(v)
    # Bins
    q25, q75 = np.percentile(vel_inst, [25, 75])
    bin_width = 2 * (q75 - q25) * len(vel_inst) ** (-1/3)
    bins = round((max(vel_inst) - min(vel_inst)) / bin_width)
    # Plot distributions
    plt.figure()
    sns.displot(vel_inst, stat=stat, bins=bins, common_norm=False, palette="bright")
    plt.title(f'Event velocity distribution: {titlestring}')
    plt.xlabel('Event velocity [nm/s]')
    plt.savefig(f'.\motor_objects\\{dirct}\\{subdir}\\figures\\{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return

### Boxplots/violin plots ###

def violin_trace_vel(dirct, figname, titlestring, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """

    dict_vel = {}
    for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            trace_velocities = []
            # Loop through lists in nested list
            for index, xb_list in enumerate(motor0.x_bead):
                xb = xb_list
                t = motor0.time_points[index]
                vel = (xb[-1]-xb[0])/(t[-1]-t[0])
                trace_velocities.append(vel)
            # append to dictionary
            dict_vel[subdir] = trace_velocities
        break

    if not os.path.isdir(f'.\motor_objects\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\\figures')
    #xb_ip_sorted = sorted(xb_ip)
    #y = np.arange(1, len(xb_ip_sorted)+1)/len(xb_ip_sorted)
    df_vel = pd.DataFrame({key:pd.Series(value) for key, value in dict_vel.items()})
    melted_vel = pd.melt(df_vel, value_vars=df_vel.columns, var_name='settings').dropna()
    # plotting
    plt.figure()
    sns.violinplot(data=melted_vel, x='settings', y='value')
    plt.ylabel('Trace velocity [nm/s]')
    plt.title(f'Bead trace velocity: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\violin_tracevel_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    return


def violin_fu_rl(dirct, k_t, figname, titlestring, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """

    dict_fu = {}
    dict_rl = {}
    for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            rl_bead = motor0.runlength_bead
            fu_bead = [i*k_t for i in rl_bead]
            # append to dictionary
            dict_fu[subdir] = fu_bead
            dict_rl[subdir] = rl_bead
        break

    if not os.path.isdir(f'.\motor_objects\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\\figures')
    #xb_ip_sorted = sorted(xb_ip)
    #y = np.arange(1, len(xb_ip_sorted)+1)/len(xb_ip_sorted)
    df_fu = pd.DataFrame({key:pd.Series(value) for key, value in dict_fu.items()})
    melted_fu = pd.melt(df_fu, value_vars=df_fu.columns, var_name='settings').dropna()

    df_rl = pd.DataFrame({key:pd.Series(value) for key, value in dict_rl.items()})
    melted_rl = pd.melt(df_rl, value_vars=df_rl.columns, var_name='settings').dropna()
    # plotting
    plt.figure()
    sns.violinplot(data=melted_fu, x='settings', y='value')
    plt.ylabel('Unbinding force [pN]')
    plt.title(f'Bead unbinding force: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\violin_fu_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    plt.figure()
    sns.violinplot(data=melted_rl, x='settings', y='value')
    plt.ylabel('Run length [nm]')
    plt.title(f'Bead run length: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\violin_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
    else:
        plt.clf()
        print('Figure saved')
    return



### CDF plots ###

def cdf_xbead(dirct, figname, titlestring, stepsize=0.001, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    dict_xb = {}
    for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            # Unpickle test_motor0
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
                interval = (0, t[-1])
                t_intrpl = np.arange(interval[0], interval[1], stepsize)
                # Do interpolation on new data points
                xb_intrpl = f(t_intrpl)
                # Remove zeroes
                xb_intrpl_nozeroes = [x for x in xb_intrpl if x != 0]
                # Remove Nans
                xb_intrpl_nonans = [x for x in xb_intrpl_nozeroes if np.isnan(x) == False]

                # Add interpolated data points to list
                xb_ip.extend(xb_intrpl_nonans)

            # append to dictionary
            dict_xb[subdir] = xb_ip

        break
    if not os.path.isdir(f'.\motor_objects\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\\figures')
    #xb_ip_sorted = sorted(xb_ip)
    #y = np.arange(1, len(xb_ip_sorted)+1)/len(xb_ip_sorted)
    df_xb = pd.DataFrame({key:pd.Series(value) for key, value in dict_xb.items()})
    melted_xb = pd.melt(df_xb, value_vars=df_xb.columns, var_name='settings').dropna()
    # plotting
    plt.figure()
    sns.ecdfplot(data=melted_xb, x='value', hue="settings")
    plt.xlabel('Bead location [nm]')
    plt.title(f'CDF of bead displacement: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\cdf_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def cdf_trace_vel(dirct, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    dict_vel = {}
    for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            trace_velocities = []
            # Loop through lists in nested list
            for index, xb_list in enumerate(motor0.x_bead):
                xb = xb_list
                t = motor0.time_points[index]
                vel = (xb[-1]-xb[0])/(t[-1]-t[0])
                trace_velocities.append(vel)
            # append to dictionary
            dict_vel[subdir] = trace_velocities
        break

    if not os.path.isdir(f'.\motor_objects\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\\figures')
    #xb_ip_sorted = sorted(xb_ip)
    #y = np.arange(1, len(xb_ip_sorted)+1)/len(xb_ip_sorted)
    df_vel = pd.DataFrame({key:pd.Series(value) for key, value in dict_vel.items()})
    melted_vel = pd.melt(df_vel, value_vars=df_vel.columns, var_name='settings').dropna()

    # plotting
    plt.figure()
    sns.ecdfplot(data=melted_vel, x='value', hue="settings")
    plt.xlabel('Trace velocity [nm/s]')
    plt.title(f'CDF of bead trace velocity: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\cdf_tracevel_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    return


def heatmap_antero_retro(dirct, figname, titlestring, show=False):
    """

    Parameters
    ----------
    dirct:
    index:
        This has to be a ordered as
    cols:


    Returns
    -------

    """
    index = ['1-1', '2-2', '3-3', '4-4', '5-5']
    columns = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    array_antero = np.zeros((len(index), len(columns)))
    array_retro = np.zeros((len(index), len(columns)))

    index_teamsize = 0
    index_ratio = 0

    for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
        for index, subdir in enumerate(subdirs):
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print(f'index_teamsize={index_teamsize}')
            print(f'index_ratio={index_ratio}')
            print(f'subdir={subdir}')
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            # Unpickle motor team
            pickle_file_motorteam = open(f'.\motor_objects\\{dirct}\{subdir}\motorteam', 'rb')
            motorteam = pickle.load(pickle_file_motorteam)
            pickle_file_motorteam.close()
            print(f'teamsize={len(motorteam)}')
            print(f'ratio={motorteam[-1].k_m/motorteam[0]}.k_m')

            #
            flattened_antero = [val for sublist in motor0.antero_motors for val in sublist]
            flattened_retro = [val for sublist in motor0.retro_motors for val in sublist]
            mean_antero = sum(flattened_antero)/len(flattened_antero)
            mean_retro = sum(flattened_retro)/len(flattened_retro)
            #
            array_antero[index_teamsize, index_ratio] = mean_antero
            array_retro[index_teamsize, index_ratio] = mean_retro

            print(array_antero)
            print(array_retro)
            #
            if index_ratio < 9:
                index_ratio += 1
            elif index_ratio == 9:
                index_ratio = 0
                index_teamsize += 1
            else:
                print('This cannot be right')

    print('done')
    print(array_antero)
    print(array_retro)
    #
    if not os.path.isdir(f'.\motor_objects\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\\figures')
    #
    df_antero = pd.DataFrame(array_antero, index=index, columns=columns)
    df_retro = pd.DataFrame(array_retro, index=index, columns=columns)
    #
    plt.figure()
    sns.heatmap(df_antero, annot=True)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f': {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\heatmap_antero_boundmotors_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    #
    plt.figure()
    sns.heatmap(df_retro, annot=True)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f': {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\heatmap_retro_boundmotors_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    return

####################


def plot_N_km_meanmaxdist(dirct, filename, figname=None, titlestring=None, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    '''
    #
    # Show the joint distribution using kernel density estimation
    plt.figure()
    g = sns.jointplot(
    data=df.loc[df.km == 0.2],
    x="meanmaxdist", y="runlength", hue="teamsize",
    kind="kde", cmap="bright")
    plt.show()
    '''
    # count plot
    km_select = ['0.1', '0.02', '0.2', 0.1, 0.2, 0.02]
    ts_select = ['2', '3', '[2]', '[3]', 2, 3, [2], [3]]
    plt.figure()
    g = sns.displot(data=df[df['km'].isin(km_select)][df['teamsize'].isin(ts_select)], x='meanmaxdist', hue='km', col='teamsize', binwidth=8, stat='count', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False)
    g.fig.suptitle(f' {titlestring}')
    g.set_xlabels(' mean max distance per run [nm]')
    g.add_legend()
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_N_km_Fex{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # hue=teamsize
    plt.figure()
    g = sns.catplot(data=df[df['km'].isin(km_select)][df['teamsize'].isin(ts_select)], x="km", y="meanmaxdist", hue="teamsize", style='teamsize', marker='teamsize', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel(' mean max dist [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def plot_N_km_fex_boundmotors(dirct, filename, figname=None, titlestring=None, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # hue=km
    plt.figure()
    g = sns.FacetGrid(data=df, hue="km", col='teamsize', style='km', marker='km', kind="point")
    g.map(sns.catplot, x="f_ex", y="boundmotors")

    g._legend.set_title('Motor stiffness [pN/nm]:')
    plt.xlabel('external force [pN]')
    plt.ylabel('<Xb> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_N_huekm_fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # count plot
    plt.figure()
    g = sns.displot(data=df, x='runlength', hue='f_ex', col='teamsize', row= 'km', stat='count', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False)
    g.fig.suptitle(f'Histogram of cargo run length {titlestring}')
    g.set_xlabels('<Xb> [nm]')
    g.add_legend()
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_runlength_N_km_fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')


    '''
    #
    # Add the joint and marginal histogram plots
    plt.figure()
    g = sns.JointGrid(data=df.loc[df.km == 0.2], x="runlength", y="boundmotors", hue='teamsize', marginal_ticks=True)
    g.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="bright")
    g.plot_marginals(sns.histplot, color="bright")
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''

    return

#########################
def plot_N_kmratio_boundmotors(dirct, filename1, filename2, figname=None, titlestring=None, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df1 = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename1}')
    df2 = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename2}')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    km_select = [1.0, 0.1, 0.5]
    sns.set_style("whitegrid")
    # hue=km
    plt.figure()
    sns.catplot(data=df1[df1['km_ratio'].isin(km_select)], x="team_size", y="antero_bound", hue="km_ratio", style='km', marker='km', kind="point")
    plt.xlabel('Team size N')
    plt.ylabel('Bound antero motors n')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_anterobound_N_kmratio_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # Boxplot
    plt.figure()
    sns.boxplot(data=df1[df1['km_ratio'].isin(km_select)], x='km_ratio', y='antero_bound', hue='team_size')
    plt.xlabel('k_m ratio')
    plt.ylabel('Bound antero motors n')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_anterobound_Nkmratio_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')


####################
    # hue=km
    plt.figure()
    sns.catplot(data=df2[df2['km_ratio'].isin(km_select)], x="team_size", y="retro_bound", hue="km_ratio", style='km_ratio', marker='km_ratio', kind="point")
    plt.xlabel('Team size N')
    plt.ylabel('Bound retro motors n')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_retrobound_N_kmratio_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # Boxplot
    plt.figure()
    sns.boxplot(data=df2[df2['km_ratio'].isin(km_select)], x='km_ratio', y='retro_bound', hue='team_size')
    plt.xlabel('k_m ratio')
    plt.ylabel('Bound retro motors n')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_retrobound_Nkmratio_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    '''
    #
    # Add the joint and marginal histogram plots
    plt.figure()
    g = sns.JointGrid(data=df.loc[df.km == 0.2], x="runlength", y="boundmotors", hue='teamsize', marginal_ticks=True)
    g.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="bright")
    g.plot_marginals(sns.histplot, color="bright")
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''

    return
def plot_N_km_runlength(dirct, filename, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    #
    xb_list = []
    v = 0.74
    N = [1,2,3,4]
    k_bind = 5
    k_unbind = 0.66
    for i in N:
        xb = v/(i*k_bind) * ( (1 + k_bind/k_unbind )**i -1 )
        xb_list.append(xb*1000)
    df2 = pd.DataFrame(columns=['N', 'xb'])
    df2['N'] = N
    df2['xb'] = xb_list
    print(df2)

    #
    sns.color_palette()
    sns.set_style("whitegrid")

    # ecdf plot
    plt.figure()
    g = sns.FacetGrid(data=df, hue='km', col="teamsize")
    g.map(sns.ecdfplot, 'runlength')
    g.add_legend(title='Motor stiffness:')
    g.fig.suptitle(f'Cumulative distribution of average cargo run length {titlestring}')
    g.set_xlabels('<Xb> [nm]')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\ecdf_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # hue=teamsize
    plt.figure()
    g = sns.catplot(data=df, x="km", y="runlength", hue="teamsize", style='teamsize', marker='teamsize', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel('<Xb> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_hueN_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    # hue=km
    plt.figure()
    g = sns.catplot(data=df, x="teamsize", y="runlength", hue="km", style='km', marker='km', kind="point")
    g._legend.set_title('Motor stiffness [pN/nm]:')
    plt.xlabel('Team size N')
    plt.ylabel('<Xb> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_N_huekm_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # Analytical included in hue=km and log
    #ax.legend(title="Motor stiffness [pN/nm]:")
    plt.figure()
    ax1 = sns.catplot(data=df, x="teamsize", y="runlength", hue="km", style='km', marker='km', kind="point")
    ax2 = sns.pointplot(data=df2, x='N', y='xb', errorbar=None, color='b', label='analytical', legend=True)
    ax1.set(yscale="log")
    ax2.set(yscale="log")
    ax1._legend.set_title('Motor stiffness [pN/nm]:')

    plt.xlabel('Team size N')
    plt.ylabel('<Xb> [nm]')
    plt.title(f': {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_N_huekm_ana_log_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # Analytical included in hue=km
    #ax.legend(title="Motor stiffness [pN/nm]:")
    fig, ax = plt.subplots()
    ax1 = sns.catplot(data=df, x="teamsize", y="runlength", hue="km", style='km', marker='km', kind="point")
    ax2 = sns.pointplot(data=df2, x='N', y='xb', errorbar=None, label='Analytical')
    ax1._legend.set_title('Motor stiffness [pN/nm]:')

    plt.xlabel('Team size N')
    plt.ylabel('<Xb> [nm]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_N_huekm_ana_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # Heatmap
    df_pivot = pd.pivot_table(df, values='runlength', index='teamsize', columns='km', aggfunc={'runlength': np.mean})
    print(df_pivot)
    plt.figure()
    sns.heatmap(df_pivot, cbar=True, cbar_kws={'label': '<Xb>'})
    plt.xlabel('motor stiffness [pN/nm]')
    plt.ylabel('teamsize N')
    plt.title(f'Heatmap of average cargo run length {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\contourplot_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # count plot
    plt.figure()
    g = sns.displot(data=df, x='runlength', hue='km', col='teamsize', stat='count', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False)
    g.fig.suptitle(f'Histogram of cargo run length {titlestring}')
    g.set_xlabels('<Xb> [nm]')
    g.add_legend()
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')


    # swarm plot
    plt.figure()
    sns.swarmplot(data=df, x="km", y="runlength", hue="teamsize", s=1, dodge=True)
    plt.xlabel('motor stiffness [pN/nm]')
    plt.ylabel('Xb [nm]')
    plt.title(f' Swarmplot of cargo run length {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\swarmplot_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # box plot
    plt.figure()
    g = sns.catplot(data=df, x="km", y="runlength", hue='teamsize',  kind='box')
    plt.xlabel('motor stiffness [pN/nm]')
    plt.ylabel('<Xb> [nm]')
    plt.title(f' Boxplot of cargo run length {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


##############################################


def plot_fex_N_km_runlength(dirct, filename, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    #
    sns.color_palette()
    sns.set_style("whitegrid")

    #
    plt.figure()
    sns.catplot(data=df, x='f_ex', y='run_length', hue='km', col='team_size', kind='box', col_wrap=2)
    plt.xlabel('External force [pN]')
    plt.ylabel('Bead run length [nm]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_rl_nfexkm_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''
    # ecdf plot
    plt.figure()
    g = sns.FacetGrid(data=df, hue='km', col="teamsize")
    g.map(sns.ecdfplot, 'runlength')
    g.add_legend(title='Motor stiffness:')
    g.fig.suptitle(f'Cumulative distribution of average cargo run length {titlestring}')
    g.set_xlabels('<Xb> [nm]')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\ecdf_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''
    '''
    # hue=km col=teamsize
    plt.figure()
    g = sns.catplot(data=df, x="f_ex", y="run_length", hue="km", col='team_size', style='team_size', marker='team_size', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel('<Xb> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\point_rl_Nfexkm_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''
    '''

    # Heatmap
    df_pivot = pd.pivot_table(df, values='runlength', index='teamsize', columns='km', aggfunc={'runlength': np.mean})
    print(df_pivot)
    plt.figure()
    sns.heatmap(df_pivot, cbar=True, cbar_kws={'label': '<Xb>'})
    plt.xlabel('motor stiffness [pN/nm]')
    plt.ylabel('teamsize N')
    plt.title(f'Heatmap of average cargo run length {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\contourplot_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # count plot
    plt.figure()
    g = sns.displot(data=df, x='runlength', hue='km', col='teamsize', stat='count', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False)
    g.fig.suptitle(f'Histogram of cargo run length {titlestring}')
    g.set_xlabels('<Xb> [nm]')
    g.add_legend()
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')


    # swarm plot
    plt.figure()
    sns.swarmplot(data=df, x="km", y="runlength", hue="teamsize", s=1, dodge=True)
    plt.xlabel('motor stiffness [pN/nm]')
    plt.ylabel('Xb [nm]')
    plt.title(f' Swarmplot of cargo run length {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\swarmplot_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # box plot
    plt.figure()
    g = sns.catplot(data=df, x="km", y="runlength", hue='teamsize',  kind='box')
    plt.xlabel('motor stiffness [pN/nm]')
    plt.ylabel('<Xb> [nm]')
    plt.title(f' Boxplot of cargo run length {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_runlength_N_km_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''
    return


################
def distplots_rl(dirct, filename, figname, titlestring, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """


    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    km_select = [1.0, 0.1, 0.5]
    sns.set_style("whitegrid")
    # hue=km
    plt.figure()
    sns.catplot(data=df[df['km_ratio'].isin(km_select)], x="team_size", y="run_length", hue="km_ratio", style='km_ratio', marker='km_ratio', kind="point")

    plt.xlabel('teamsize')
    plt.ylabel('<run length> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    plt.figure()
    sns.boxplot(data=df[df['km_ratio'].isin(km_select)], x='km_ratio', y='run_length', hue='team_size')
    plt.xlabel('k_m ratio')
    plt.ylabel('Bead run length [nm]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # plotting
    plt.figure()
    sns.displot(data=df, x='rl', hue='km_ratio', col='teamsize', stat='probability', multiple='stack', common_norm=False, palette='bright')
    plt.xlabel('Bead run length [nm]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')



    return


def distplots_xb(dirct, filename, figname, titlestring, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """


    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    plt.figure()
    sns.boxplot(data=df, x='km_ratio', y='xb', hue='teamsize')
    plt.xlabel('k_m ratio')
    plt.ylabel('Bead displacement [nm]')
    plt.title(f' Distribution displacement: {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # plotting
    plt.figure()
    sns.displot(df, x='xb', hue='km_ratio', col='teamsize', stat='probability', multiple='stack', common_norm=False)
    plt.xlabel('Bead displacement [nm]')
    plt.title(f' Distribution displacement: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    # plotting
    plt.figure()
    sns.ecdfplot(df, x='xb', hue='km_ratio', col='teamsize', common_norm=False)
    plt.xlabel('Bead displacement [nm]')
    plt.title(f' Distribution displacement: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\ecdf_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return









