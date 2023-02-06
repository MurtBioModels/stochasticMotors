import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os
from motorgillespie.analysis import segment_trajectories as st


### F_ex and Kt ###
'''Cargo RL pdf, cdf and lineplot/barplot <>'''
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
            flat_rl = [element for sublist in runlength for element in sublist]
            print(flat_rl)
            #
            dict_rl[key] = flat_rl
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
            else:
                print('Something is wrong with parameter counting')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_rl.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='run_length', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_rl.csv')

    return
def plot_n_fex_km_rl(dirct, filename, figname, titlestring, show=False):
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
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0', '-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        #
        plt.figure()
        sns.catplot(data=df4, x="km", y="run_length", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se').set_title(f'External force = {i}pN {titlestring}')
        plt.xlabel('Trap stiffness [pN/nm]')
        plt.ylabel('<cargo run length> [nm]')
        #plt.title(f'Team size N = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_rl_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')

        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
'''Cargo displacement: pdf and cdf'''
def xb_n_fex_km(dirct, filename, ts_list, fex_list, km_list, stepsize=0.01):
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
            # Unpickle motor_0 object
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
            else:
                print('Something is wrong with parameter counting')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    #
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_xb.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='xb', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_kmratio_xb.csv')

    return
def plot_n_fex_km_xb(dirct, filename, figname, titlestring, stat='probability', show=True):
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

    ### Plotting ###
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0','-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        print('Making figure..')
        plt.figure()
        sns.displot(df4,  hue='km', row='team_size', stat=stat).set_title(f'External force = {i}pN {titlestring}')
        #plt.title(f'Distribution (interpolated) of cargo location {titlestring} ')
        plt.xlabel('Distance traveled [nm]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_xb_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
'''Cargo trajectory examples > + motor trajectories ??'''
def traj_fex(dirct, subdir, figname, titlestring, it=0, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """

    # Unpickle motor0 object
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    motor0.time_points[it].pop()
    x = motor0.time_points[it]
    y = motor0.x_bead[it]
    f_ex = motor0.f_ex
    plt.step(x, y, where='post')
    #plt.scatter(x,y)
    plt.title(f'Example of trajectory: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_{f_ex}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return
'''bound motors barplot'''
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
            else:
                print('Something is wrong with parameter counting')

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
# ADD STATS!!!
def plot_n_fex_km_boundmotors(dirct, filename, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Return
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0', '-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        print('Making figure..')
        plt.figure()
        sns.barplot(data=df4, x="team_size", y="bound_motors", hue="km", ci=95).set_title(f'External force = {i}pN {titlestring}')
        plt.xlabel('Team Size N')
        plt.ylabel('Bound Motors n')
        #plt.title(f' External force = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_boundmotors_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return # #
'''(unbinding events)'''
def unbindevent_bead_n_fex_km(dirct, filename, ts_list, fex_list, km_list):
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
    dict_ubinding = {}
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
            antero_unbinds = list(motor0.antero_unbinds)
            retro_unbinds = list(motor0.retro_unbinds)
            print(np.unique(retro_unbinds))
            #
            dict_ubinding[key] = antero_unbinds
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
            else:
                print('Something is wrong with parameter counting')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_ubinding.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='unbinding events/100s', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_rl.csv')

    return
# ADD STATS!!!
def plot_n_fex_km_unbindevent(dirct, filename, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Return
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0', '-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        print('Making figure..')
        plt.figure()
        sns.barplot(data=df4, x="team_size", y="bound_motors", hue="km", ci=95).set_title(f'External force = {i}pN {titlestring}')
        plt.xlabel('Team Size N')
        plt.ylabel('Unbinding events')
        #plt.title(f' External force = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_unbindingevents_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
'''Bind time cargo >> combine with RL? >>> or velocity??'''
def bt_bead_n_fex_km(dirct, filename, ts_list, fex_list, km_list):
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
    dict_tb = {}
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
            time_bind = list(motor0.time_unbind)
            flat_tb = [element for sublist in time_bind for element in sublist]
            #
            dict_tb[key] = flat_tb
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
            else:
                print('Something is wrong with parameter counting')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_tb.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='time_bind', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}_N_fex_km_tb.csv')

    return
def plot_n_fex_km_bt(dirct, filename, figname, titlestring, show=False):
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
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0', '-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        #
        plt.figure()
        sns.catplot(data=df4, x="km", y="time_bind", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se').set_title(f'External force = {i}pN {titlestring}')
        plt.xlabel('Trap stiffness [pN/nm]')
        plt.ylabel('<Time bound> [s]')
        #plt.title(f'Team size N = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_tb_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')

        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
'''segments or x vs v segments >> eventueel RL opsplitsen en dan segmenten'''
def segment_parratio_asc(dirct, filename, ts_list, fex_list, km_list):
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
    nested_seg = []
    key_tuples = []
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
            print(os.path.join(path,subdir))
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
            xb = motor0.x_bead
            t = motor0.time_points
            del motor0
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
            runs_asc, t_asc = st.diff_asc(x_list=xb, t_list=t)
            flat_runs_asc = [element for sublist in runs_asc for element in sublist]
            #
            nested_seg.append(tuple(flat_runs_asc))

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
            else:
                print('Something is wrong with parameter counting')

    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_seg)}')
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
    df1 = pd.DataFrame(nested_seg[:mid_index], index=multi_column[:mid_index])
    print(df1)
    df2 = pd.DataFrame(nested_seg[mid_index:], index=multi_column[mid_index:])
    print(df2)
    del nested_seg
    df3 = pd.concat([df1, df2]).T
    del df1
    del df2
    #print(df3)
    print('Melt dataframe... ')
    df_melt = pd.melt(df3, value_name='seg_asc', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\asc_x_v{filename}.csv')

    return
def segment_parratio_desc(dirct, filename, ts_list, fex_list, km_list):
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
    nested_seg = []
    key_tuples = []
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
            print(os.path.join(path,subdir))
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
            xb = motor0.x_bead
            t = motor0.time_points
            del motor0
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
            runs_desc, t_desc = st.diff_desc(x_list=xb, t_list=t)
            flat_runs_desc = [element for sublist in runs_desc for element in sublist]
            #
            nested_seg.append(tuple(flat_runs_desc))

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
            else:
                print('Something is wrong with parameter counting')

    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_seg)}')
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
    df1 = pd.DataFrame(nested_seg[:mid_index], index=multi_column[:mid_index])
    print(df1)
    df2 = pd.DataFrame(nested_seg[mid_index:], index=multi_column[mid_index:])
    print(df2)
    del nested_seg
    df3 = pd.concat([df1, df2]).T
    del df1
    del df2
    #print(df3)
    print('Melt dataframe... ')
    df_melt = pd.melt(df3, value_name='seg_desc', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\desc_x_v{filename}.csv')

    return
# finish!!!
def plot_n_fex_km_seg(dirct, filename1, filename2, figname, titlestring, stat='probability', show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    df_asc = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename1}')
    df_desc = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename2}')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0','-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        print('Making figure...')
        plt.figure()
        sns.displot(df4,  hue='km', col='team_size', stat=stat).set_title(f'External force = {i}pN {titlestring}')
        #plt.title(f'Distribution (interpolated) of cargo location {titlestring} ')
        plt.xlabel('Segmented runs [nm]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_seg_asc_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

## motors ##
'''Motor forces pdf'''
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
def plot_fex_N_km_forces_motors(dirct, filename1=None, filename2=None, figname='', titlestring='', total=True, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    if total == False:
        df1 = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename1}')
        df1_nozeroes = df1[df1['motor_forces'] != 0 ]
        df2 = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename2}')
        df2_nozeroes = df2[df2['motor_forces'] != 0 ]
        df_total = pd.concat([df1_nozeroes, df2_nozeroes], ignore_index=True)
        df_total.to_csv(f'.\motor_objects\\{dirct}\\data\\N_fex_km_motorforces_total.csv')
    else:
        df_total = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\N_fex_km_motorforces_total.csv')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.set_style("whitegrid")

    fexlist = [-4, -5, -6, -7]
    #

    for i in fexlist:
        df2 = df_total[df_total['f_ex'] == i]
        print(df2)
        #forces = list(df3['motor_forces'])

        # Bins
        #q25, q75 = np.percentile(forces, [25, 75])
        #bin_width = 2 * (q75 - q25) * len(forces) ** (-1/3)
        #bins_forces = round((max(forces) - min(forces)) / bin_width)
        plt.figure()
        sns.displot(df2, x='motor_forces', col='team_size', stat='probability',binwidth=0.2, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Motor forces [pN]')
        plt.title(f' Distribution motor forces fex={i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_fmotors_n_fex_km_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    #
    '''
    for i in teamsize_select:
        df2 = df_total[df_total['team_size'] == i]
        df3 = df2[df2['km'].isin(km_select)]
        print(df3)
        #forces = list(df3['motor_forces'])

        # Bins
        #q25, q75 = np.percentile(forces, [25, 75])
        #bin_width = 2 * (q75 - q25) * len(forces) ** (-1/3)
        #bins_forces = round((max(forces) - min(forces)) / bin_width)
        plt.figure()
        sns.displot(df3, x='motor_forces', hue='f_ex', col='km', stat='probability',binwidth=0.2, palette='bright', common_norm=False, common_bins=True)
        plt.xlabel('Motor forces [pN]')
        plt.title(f' Distribution motor forces N={i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_fmotors_n_fex_km_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')
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
'''Motor displacement pdf'''
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
def plot_fex_N_km_xm(dirct, filename, figname='', titlestring='', show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #

    df_total = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.set_style("whitegrid")

    fexlist =  [-4, -5, -6, -7]
    #

    for i in fexlist:
        df2 = df_total[df_total['f_ex'] == i]
        print(df2)
        #forces = list(df3['motor_forces'])

        # Bins
        #q25, q75 = np.percentile(forces, [25, 75])
        #bin_width = 2 * (q75 - q25) * len(forces) ** (-1/3)
        #bins_forces = round((max(forces) - min(forces)) / bin_width)
        plt.figure()
        sns.displot(df2, x='xm', col='team_size', stat='probability',binwidth=0.2, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Distance [nm]')
        plt.title(f' Distribution Xm fex={i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_xm_n_fex_km_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    #
    '''
    for i in teamsize_select:
        df2 = df_total[df_total['team_size'] == i]
        df3 = df2[df2['km'].isin(km_select)]
        print(df3)
        #forces = list(df3['motor_forces'])

        # Bins
        #q25, q75 = np.percentile(forces, [25, 75])
        #bin_width = 2 * (q75 - q25) * len(forces) ** (-1/3)
        #bins_forces = round((max(forces) - min(forces)) / bin_width)
        plt.figure()
        sns.displot(df3, x='motor_forces', hue='f_ex', col='km', stat='probability',binwidth=0.2, palette='bright', common_norm=False, common_bins=True)
        plt.xlabel('Motor forces [pN]')
        plt.title(f' Distribution motor forces N={i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_fmotors_n_fex_km_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')
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
'''Motor RL pdf, cdf and lineplot/barplot <>'''
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
def plot_fex_N_km_fu_motors(dirct, filename, figname, titlestring, show=False):
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
    km_select = [0.02, 0.1, 0.2]
    n_select = [1, 2, 3, 4, 5]
    #
    for i in n_select:
        df2 = df[df['team_size'].isin([i])]
        df3 = df2[df2['km_'].isin(km_select)]
        plt.figure()
        sns.catplot(data=df3, x='f_ex', y='fu_motors', hue='km_', kind='box')
        plt.xlabel('External force [pN]')
        plt.ylabel('Motor unbinding force [pN]')
        plt.title(f' Teamsize = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_fu_motor_nfexkm_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')
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
        sns.catplot(data=df3, x="f_ex", y="fu_motors", hue="km_", style='km_', marker='km_', kind="point")
        plt.xlabel('teamsize')
        plt.ylabel('<motor unbinding force> [pN]')
        plt.title(f'Teamsize = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_fu_motors_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')

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
'''(((contourplots??)))'''

### Symmetry ###
'''Cargo RL pdf, cdf and lineplot/barplot <>'''
def rl_cargo_n_kmr(dirct, filename, ts_list, kmminus_list):
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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_km_minus_rl.csv')

    return
def plot_n_kmratio_rl(dirct, filename, figname, titlestring, show=True):
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

    km_select = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    #df2 = df[df['km_minus'].isin(km_select)]

    #
    print(f'Start plotting...')
    plt.figure()
    sns.catplot(data=df, x="km_minus", y="run_length", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se')
    plt.xlabel('Trap stiffness of minus motor [pN/nm]')
    plt.ylabel('<Cargo run length> [nm]')
    plt.title(f'{titlestring}')
    print(f'Start saving...')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_rl_{figname}.png', format='png', dpi=300)
    # bbox_inches='tight'
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return
'''Cargo displacement: pdf and cdf'''
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
    nested_xb = []
    key_tuples = []
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
            key_tuples.append(key)
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
            nested_xb.append(tuple(xb_total_interpl))

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
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_xb)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_ratio'])
    print(multi_column)
    del key_tuples
    #
    mid_index = 20
    print(f'mid_index={mid_index}')
    print(f'multi_column[:mid_index]={multi_column[:mid_index]}')
    print(f'multi_column[mid_index:]={multi_column[mid_index:]}')
    #
    df1 = pd.DataFrame(nested_xb[:mid_index], index=multi_column[:mid_index]).T
    print(df1)
    df2 = pd.DataFrame(nested_xb[mid_index:], index=multi_column[mid_index:]).T
    print(df2)
    del nested_xb
    #df3 = pd.concat([df1, df2]).T
    #del df1
    #del df2
    #print(df3)

    print('Melt dataframes_figures... ')
    df_melt1 = pd.melt(df1, value_name='xb', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt1)
    del df1
    df_melt2 = pd.melt(df2, value_name='xb', var_name=['team_size', 'km_ratio']).dropna()
    print(df_melt2)
    del df2
    #
    print('Save dataframe... ')
    df_melt1.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_xb1.csv')
    df_melt2.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_xb2.csv')

    return
def plot_N_kmr_xb(dirct, filename, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    ### Plotting ###
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    n_select = ['2', '3', '4']
    df2 = df[df['team_size'].isin(n_select)]
    km_select = ['0.1', '0.12', '0.14', '0.16', '0.18', '0.2']
    df3 = df2[df2['km'].isin(km_select)]
    fex_select = ['0', '-1', '-2', '-3', '-4', '-5', '-6', '-7']

    #
    for i in fex_select:

        df4 = df3[df3['f_ex'].isin([i])]
        print('Making figure..')
        plt.figure()
        sns.displot(df4,  hue='km', row='team_size', stat=stat).set_title(f'External force = {i}pN {titlestring}')
        #plt.title(f'Distribution (interpolated) of cargo location {titlestring} ')
        plt.xlabel('Distance traveled [nm]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_xb_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df_nozeroes = df[df['xb'] != 0]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.set_style("whitegrid")
    km_select = [0.1, 0.5, 1]
    teamsize_select = [str([1, 1]), str([2, 2]), str([3, 3]), str([4, 4])]

    '''
    #
    for i in km_select:
        df2 = df_nozeroes[df_nozeroes['km_ratio'] == i]
        df3 = df2[df2['team_size'].isin(teamsize_select)]
        print(df3)
        #forces = list(df3['motor_forces'])

        # Bins
        #q25, q75 = np.percentile(forces, [25, 75])
        #bin_width = 2 * (q75 - q25) * len(forces) ** (-1/3)
        #bins_forces = round((max(forces) - min(forces)) / bin_width)
        # plotting
        plt.figure()
        sns.displot(df3, x='xb', col='team_size', stat='probability', binwidth=4, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Displacement [nm]')
        plt.title(f' Distribution displacement cargo: {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_xb_colN_{figname}_{i}kmr.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    #
    for i in teamsize_select:
        df2 = df_nozeroes[df_nozeroes['team_size'] == i]
        df3 = df2[df2['km_ratio'].isin(km_select)]
        print(df3)
        #forces = list(df3['motor_forces'])

        # Bins
        #q25, q75 = np.percentile(forces, [25, 75])
        #bin_width = 2 * (q75 - q25) * len(forces) ** (-1/3)
        #bins_forces = round((max(forces) - min(forces)) / bin_width)
        plt.figure()
        sns.displot(df3, x='xb', col='km_ratio', stat='probability', binwidth=0.2, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Displacement [nm]')
        plt.title(f' Distribution displacement cargo {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_xb_colkmr_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')
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
    plt.figure()
    sns.catplot(data=df[df['km_ratio'].isin(km_select)], x="team_size", y="xb", hue="km_ratio", style='km_ratio', marker='km_ratio', kind="point")
    plt.xlabel('teamsize')
    plt.ylabel(' <bead displacement> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_xb_nozeroes_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    '''
    plt.figure()
    sns.boxplot(data=df[df['km_ratio'].isin(km_select)], x='km_ratio', y='xb', hue='team_size')
    plt.xlabel('teamsize')
    plt.ylabel(' <bead displacement> [nm]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return
'''Cargo trajectory examples > + motor trajectories ??'''

'''bound motors barplot'''
# ADD STATS!!!

'''(unbinding events)'''
# ADD STATS!!!

'''Bind time cargo >> combine with RL? >>> or velocity??'''
'''segments or x vs v segments >> eventueel RL opsplitsen en dan segmenten'''
# finish!!!


## motors ##
'''Motor forces pdf'''
'''Motor displacement pdf'''
'''Motor RL pdf, cdf and lineplot/barplot <>'''
'''(((contourplots??)))'''
