import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os
from motorgillespie.analysis import segment_trajectories as st
import random

# index=false in dataframes!!

##### ELASTIC COUPLING, N + FEX + KM #####
'''Cargo RL pointplot''' #REPORT #DONE
def rl_bead_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : str
            Name of the subdirectory within the `motor_object` directory
    ts_list : list of int or list of float
                  Values of the motor_team size N within `dirct`
    fex_list : list of int or list of float
                  Values of the external force f_ex within `dirct`
    km_list : list of int or list of float
                  Values of the motor stiffness km within `dirct`
    filename : str, optional
            Addition to include into the default file name

    Returns
    -------
    None

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
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            #
            runlength = list(motor0.runlength_bead) #runlength_cargo
            flat_rl = [element for sublist in runlength for element in sublist]
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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\rl_cargo_N_fex_km_{filename}.csv', index=False)

    return
def plot_n_fex_km_rl(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : list of int or list of float
                  Which values of the motor_team size N to include
    fex_include : list of int or list of float
                  Which values of the external force f_ex to include
    km_include : list of int or list of float
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['team_size'].isin(n_include)]
    df3 = df2[df2['f_ex'].isin(fex_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #print(df4)
    #df4['run_length'] = df['run_length'].div(1000).round(2)
    #print(df4)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        plt.figure()
        g = sns.catplot(data=df_fex, x="km", y="run_length", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se')
        plt.title(r'$F_{ex}= $' f'{i}pN')
        g._legend.set_title('External force [pN]')
        plt.xlabel(r'$k_{m}$ [pN/nm]')
        plt.ylim(0, 21000)
        #[\u03bcm]
        plt.ylabel('Cargo Runlength [nm]')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}fex_{figname}', format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        print(df_n)
        plt.figure()
        g = sns.catplot(data=df_n, x="km", y="run_length", hue="f_ex", style='f_ex', marker='f_ex', kind="point", errornar='se')
        g._legend.set_title('External force [pN]')
        plt.title(f'{i} motors')
        plt.xlabel(r'$k_{m}$ [pN/nm]')
        plt.ylim(0, 21000)
        plt.ylabel('Cargo Runlength [nm]')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}N_ylim_{figname}', format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}N_ylim_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        print(df_n)
        plt.figure()
        g = sns.catplot(data=df_n, x="km", y="run_length", hue="f_ex", style='f_ex', marker='f_ex', kind="point", errornar='se')
        g._legend.set_title('External force [pN]')
        plt.title(f'{i} motors')
        plt.xlabel(r'$k_{m}$ [pN/nm]')
        plt.ylim(0, None)
        plt.ylabel('Cargo Runlength [nm]')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
def plot_n_fex_km_rl_log(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : list of int or list of float
                  Which values of the motor_team size N to include
    fex_include : list of int or list of float
                  Which values of the external force f_ex to include
    km_include : list of int or list of float
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['team_size'].isin(n_include)]
    df3 = df2[df2['f_ex'].isin(fex_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #print(df4)
    #df4['run_length'] = df['run_length'].div(1000).round(2)
    #print(df4)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        print(df_n)
        plt.figure()
        g = sns.catplot(data=df_n, x="km", y="run_length", hue="f_ex", style='f_ex', marker='f_ex', kind="point", errornar='se')
        g._legend.set_title('External force [pN]')
        plt.title(f'{i} motors')
        plt.xlabel(r'$k_{m}$ [pN/nm]')
        plt.yscale('log')
        plt.ylim( 10**0, 10**5 )
        plt.ylabel('Cargo Runlength [nm]')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}N_log_ylim_{figname}', format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_rl_{i}N_log_ylim_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

# simulatie net km=0 ??:
def plot_n_km_rl_analytical(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['f_ex'] == 0]
    df3 = df2[df2['team_size'].isin(n_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    k_on = 5
    k_off = 0.66
    v = 740
    N = np.array(1, 2, 3, 4)
    rl = v/(N*k_on)( (1+(k_on/k_off))**N - 1 )
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    plt.figure()
    sns.catplot(data=df4, x="team_size", y="run_length", hue="km", style='team_size', marker='team_size', kind="point", errornar='se').set_title(f'External force = {i}pN')
    plt.xlabel('Trap stiffness [pN/nm]')
    plt.ylabel('<cargo run length> [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_rl_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')

    #
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return

'''Cargo bind time pointplot'''
def bt_bead_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

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
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            #
            time_bind = list(motor0.time_unbind)
            diff_tb = [list(np.diff(sublist)) for sublist in time_bind]
            print(len(diff_tb))
            flat_diff_tb = [element for sublist in diff_tb for element in sublist]
            #
            dict_tb[key] = flat_diff_tb
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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\bt_N_fex_km_{filename}.csv')

    return
def plot_n_fex_km_bt(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['team_size'].isin(n_include)]
    df3 = df2[df2['f_ex'].isin(fex_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        df_fex = df4[df4['f_ex'] == i]
        plt.figure()
        sns.catplot(data=df_fex, x="km", y="time_bind", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se')
        plt.title(f'F_ex {i}pN')
        plt.xlabel('k [pN/nm]')
        plt.ylabel('Cargo Bindtime [s]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_bt_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        plt.figure()
        sns.catplot(data=df_n, x="km", y="time_bind", hue="f_ex", style='f_ex', marker='f_ex', kind="point", errornar='se')
        plt.title(f'{i} motors')
        plt.xlabel('k [pN/nm]')
        plt.ylabel('Cargo Bindtime [s]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_bt_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

'''Cargo velocity pointplot''' #REPORT #DONE
def vel_bead_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    #
    dict_vel = {}
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
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            #
            time_bind = list(motor0.time_unbind)
            #
            run_length = list(motor0.runlength_bead) #runlength_cargo
            #
            vel_flat = []
            for i, sub_rl in enumerate(run_length):
                print(f'len(sub_rl)={len(sub_rl)}')
                print(f'len(time_bind[i])={len(time_bind[i])}')
                #print(time_bind[i])
                time_bind[i].insert(0,0)
                #print(time_bind[i])
                diff_tb = np.diff(np.asarray(time_bind[i]))
                #print(diff_tb)
                if len(sub_rl) != len(diff_tb):
                    raise AssertionError('There should be equal rl as unbindtime values')
                list_vel = [rl/tb for rl,tb in zip(sub_rl, diff_tb)]
                vel_flat.extend(list_vel)

            #
            dict_vel[key] = vel_flat
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

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_vel.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='velocity_cargo_run', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\vel_N_fex_km_{filename}.csv')

    return
def plot_n_fex_km_vel(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : bool,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['team_size'].isin(n_include)]
    df3 = df2[df2['f_ex'].isin(fex_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')

    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        plt.figure()
        g = sns.catplot(data=df_n, x="km", y="velocity_cargo_run", hue="f_ex", style='f_ex', marker='f_ex', kind="point", errornar='se')
        g._legend.set_title('External force [pN]')
        plt.title(f'{i} motors')
        plt.xlabel(r'$k_{m}$ [pN/nm]')
        plt.ylim(0, 700)
        plt.ylabel('Cargo velocity [nm/s]')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_vel_{i}N_ylim_{figname}', format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_vel_{i}N_ylim_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

'''Cargo displacement pdf'''
def xb_n_fex_km(dirct, ts_list, fex_list, km_list, stepsize=0.1, samplesize=100, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_xb = []
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
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'fex_count={fex_count}')
            print(f'km_count={km_count}')
            # Unpickle motor_fixed object
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
            key_tuples.append(key)
            #
            xb = motor0.x_bead #x_cargo
            time = motor0.time_points
            del motor0
            xb_total_interpl = []
            print('Start interpolating bead locations...')
            for index, list_xb in enumerate(xb):
                    #print(f'index={index}')
                    # Original data
                    t_i = time[index]
                    xb_i = list_xb
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
                    if len(t_i) != len(xb_i):
                        t_i.pop()
                    # Create function
                    f = interp1d(t_i, xb_i, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t_i[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    xb_intrpl = f(t_intrpl)
                    # Random sampling
                    xb_sampled = random.sample(list(xb_intrpl), samplesize)
                    # Add to list
                    xb_total_interpl.extend(xb_sampled)

            #
            nested_xb.append(tuple(xb_total_interpl))

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
    print(f'len(nested_nested_xb) should be {len(key_tuples)}: {len(nested_xb)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples
    #
    df1 = pd.DataFrame(nested_xb, index=multi_column).T
    print(df1)
    del nested_xb
    #
    print('Melt dataframe... ')
    df_melt1 = pd.melt(df1, value_name='xb', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt1)
    del df1
    #
    print('Save dataframe... ')
    df_melt1.to_csv(f'.\motor_objects\\{dirct}\\data\\xb_sampled_{filename}.csv', index=False)


    return
def plot_n_fex_km_xb(dirct, filename, n_include, fex_include, km_include, stat='probability', show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            print(j)
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            for k in km_include:
                print(k)
                df_km = df_n[round(df_n['km'], 2) == round(k, 2)]
                print(df_km)
                #
                q25, q75 = np.percentile(list(df_km['xb']), [25, 75])
                print(q25)
                print(q75)
                bin_width = 2 * (q75 - q25) * len(list(df_km['xb'])) ** (-1/3)
                print(bin_width)
                if bin_width > 0:
                    max = df_km['xb'].max()
                    print(f'max={max}')
                    min = df_km['xb'].min()
                    print(f'min={min}')
                    bins_xb = round((float(df_km['xb'].max()) - float(df_km['xb'].min())) / float(bin_width))
                else:
                    bins_xb = round(len(list(df_km['xb']))/100)
                plt.figure()
                sns.displot(df_km, x='xb', stat=stat, bins=bins_xb)
                plt.title(f'F_ex={i}pN, N={j} motors, km={k}pN/nm')
                plt.xlabel('xb [nm]')
                plt.savefig(f'.\motor_objects\\{dirct}\\figures\dist_xb_{i}fex{j}N{k}km_{figname}.png', format='png', dpi=300, bbox_inches='tight')
                if show == True:
                    plt.show()
                    plt.clf()
                    plt.close()
                else:
                    plt.clf()
                    plt.close()
                    print('Figure saved')

    return
#REPORT #DONE
def plot_n_fex_km_xb_cdf(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            print(j)
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            plt.figure()
            g = sns.ecdfplot(data=df_n, x="xb", hue="km",  palette='bright')
            #g._legend.set_title(r'k_{m} [pN/nm]')
            plt.title(f'{j} motors,' r'$F_{ex}= $' f'{i}pN')
            plt.xlabel('Cargo displacement [nm]')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\ecdf_xb_{i}fex{j}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\ecdf_xb_{i}fex{j}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return

'''Cargo trajectory examples'''
def traj_n_fex_km(dirct, ts_list, fex_list, km_list, show=True):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed

    Returns
    -------
    None

    """

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
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
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            random_index = random.sample(range(0, 999), 5)
            for i in random_index:
                motor0.time_points[i].pop()
                x = motor0.time_points[i]
                y = motor0.x_bead[i] #x_cargo
                # Plotting
                if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
                    os.makedirs(f'.\motor_objects\\{dirct}\\figures')
                sns.color_palette()
                sns.set_style("whitegrid")
                print('Making figure..')
                plt.step(x, y, where='post')
                #plt.scatter(x,y)
                plt.title(f'Example of trajectory: N={ts} F_ex={fex} Km minus={km}')
                plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_{ts}_{fex}_{km}_{i}.png', format='png', dpi=300, bbox_inches='tight')
                if show == True:
                    plt.show()
                    plt.clf()
                    plt.close()
                else:
                    plt.clf()
                    plt.close()
                    print('Figure saved')

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

    return

'''bound motors barplot'''
def bound_n_fex_km(dirct, ts_list, fex_list, km_list, stepsize=0.01, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
              Values of the motor_team size N within dirct
    fex_list : list of floats,
                Values of the external force f_ex within dirct
    km_list : list of floats,
              Values of the motor stiffness km within dirct
    stepsize : float, default=0.01
               Step size for interpolation function
    filename : string, optional
               Addition to include into the default file name

    Returns
    -------
    None

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_bound = []
    keys_bound = []
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
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            keys_bound.append(key)
            #
            retro = motor0.retro_motors #retro_bound
            if retro != 0:
                print(f'There should be no minus motors, check simulation settings')
            #
            motors_bound = motor0.antero_motors #antero_bound
            mean_bound_motors = []
            #
            print('Start interpolating bound motors...')
            for index, list_bm in enumerate(motors_bound):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    bound = list_bm
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
            nested_bound.append(mean_bound_motors)
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
    print(f'len(nested_nested_xb) should be {len(nested_bound)}: {len(nested_bound)}')
    #
    multi_column_antero = pd.MultiIndex.from_tuples(keys_bound, names=['team_size', 'f_ex', 'km'])
    print(multi_column_antero)
    del keys_bound
    #
    df = pd.DataFrame(nested_bound, index=multi_column_antero).T
    print(df)
    del nested_bound
    #
    print('Melt antero dataframe... ')
    melt_df = pd.melt(df, value_name='bound_motors', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(melt_df)
    del df
    #
    melt_df.to_csv(f'.\motor_objects\\{dirct}\\data\\boundmotors_N_fex_km_{filename}.csv')


    return
# ADD STATS!!!
def plot_n_fex_km_boundmotors(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['team_size'].isin(n_include)]
    df3 = df2[df2['f_ex'].isin(fex_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        print('Making figure..')
        plt.figure()
        sns.barplot(data=df_n, x="f_ex", y="bound_motors", hue="km", ci=95)
        plt.xlabel('External force [pN]')
        plt.ylabel('Bound Motors n')
        plt.title(f' {i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_boundmotors_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

'''Unbind events barplot''' #REPORT #DONE
def unbindevent_bead_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    #
    nested_unbind = []
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
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            #
            retro = motor0.retro_unbinds #retro_unbind_events
            if retro != 0:
                print(f'There should be no minus motors, check simulation settings')
            #
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            key_tuples.append(key)
            nested_unbind.append(list(motor0.antero_unbinds)) #antero_unbind_events

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
    print(f'len(nested_unbind) should be {len(key_tuples)}: {len(nested_unbind)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples
    #
    df = pd.DataFrame(nested_unbind, index=multi_column).T
    print(df)
    del nested_unbind

    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='unbind_events', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\unbindevents_N_fex_km_{filename}.csv')


    return
# ADD STATS!!!
def plot_n_fex_km_unbindevent(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['team_size'].isin(n_include)]
    df3 = df2[df2['f_ex'].isin(fex_include)]
    df4 = df3[df3['km'].isin(km_include)]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
    #
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:

        df_n = df4[df4['team_size'] == i]
        plt.figure()
        g = sns.barplot(data=df_n, x="f_ex", y="unbind_events", hue="km", ci=95)
        g.legend(title= 'External force [pN]')
        sns.move_legend(g, "upper right")
        plt.xlabel('External force [pN]')
        plt.ylabel('Unbinding events n')
        plt.ylim(0,950)
        plt.yticks(np.arange(0, 950, step=100))
        plt.title(f' {i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_unbindingevents_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_unbindingevents_{i}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

'''segments pdf or x vs v segments >> eventueel RL opsplitsen en dan segmenten'''
def segment_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : str
            Name of the subdirectory within the `motor_object` directory
    ts_list : list of int or list of float
                  Values of the motor_team size N within `dirct`
    fex_list : list of int or list of float
                  Values of the external force f_ex within `dirct`
    km_list : list of int or list of float
                  Values of the motor stiffness km within `dirct`
    filename : str, optional
            Addition to include into the default file name

    Returns
    -------
    None

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
            print(f'subdir={subdir}')
            #
            print(f'teamsize_count={teamsize_count}')
            print(f'km_count={km_count}')
            print(f'fex_count={fex_count}')
            # Unpickle motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            xb = motor0.x_bead #x_cargo
            t = motor0.time_points
            del motor0
            #
            ts = ts_list[teamsize_count]
            fex = fex_list[fex_count]
            km = km_list[km_count]
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            ##### key tuple #####
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            key_tuples.append(key)
            #
            runs_asc, t_asc = st.diff_asc(x_list=xb, t_list=t)
            flat_runs_asc = [element for sublist in runs_asc for element in sublist]
            nested_seg.append(flat_runs_asc)
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
    len_tuples = len(key_tuples)
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples
    #
    mid_index = int(len_tuples/2)
    print(f'mid_index={mid_index}')
    print(f'multi_column[:mid_index]={multi_column[:mid_index]}')
    print(f'multi_column[mid_index:]={multi_column[mid_index:]}')
    #
    df1 = pd.DataFrame(nested_seg[:mid_index], index=multi_column[:mid_index]).T
    print(df1)
    print('Melt dataframe 1... ')
    df1_melt = pd.melt(df1, value_name='segments', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df1_melt)
    del df1
    df2 = pd.DataFrame(nested_seg[mid_index:], index=multi_column[mid_index:]).T
    print(df2)
    print('Melt dataframe 2... ')
    df2_melt = pd.melt(df2, value_name='segments', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df2_melt)
    del df2
    df3 = pd.concat([df1_melt, df2_melt], axis=0)
    del df1_melt, df2_melt
    print(df3)
    #
    df3.to_csv(f'.\motor_objects\\{dirct}\\data\\segments_{filename}.csv', index=False)

    return
def plot_n_fex_km_seg(dirct, filename, n_include, fex_include, km_include, stat='probability', show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : list of int or list of float
                  Which values of the motor_team size N to include
    fex_include : list of int or list of float
                  Which values of the external force f_ex to include
    km_include : list of int or list of float
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            for k in km_include:
                df_km = df_n[round(df_n['km'], 2) == round(k, 2)]
                print(df_km)

                plt.figure()
                sns.displot(df_km, x='segments', stat=stat)
                plt.title(f'F_ex={i}pN, N={j} motors, km={k}pN/nm')
                plt.xlabel('Segmented runs [nm]')
                plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_segments_{i}fex{j}N{k}km_{figname}.png', format='png', dpi=300, bbox_inches='tight')
                if show == True:
                    plt.show()
                    plt.clf()
                    plt.close()
                else:
                    plt.clf()
                    plt.close()
                    print('Figure saved')

    return

#not finished
def segment_back_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """

    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

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
            xb = motor0.x_cargo
            del motor0
            #
            ts = ts_list[teamsize_count]
            fex = fex_list[fex_count]
            km = km_list[km_count]
            print(f'ts={ts}')
            print(f'fex={fex}')
            print(f'km={km}')
            ##### key tuple #####
            key = (str(ts), str(fex), str(km))
            print(f'key={key}')
            key_tuples.append(key)
            #
            runs_back = st.diff_unbind(x_list=xb)
            flat_runs_back = [element for sublist in runs_back for element in sublist]
            nested_seg.append(flat_runs_back)
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
    len_tuples = len(key_tuples)
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples
    #
    mid_index = len_tuples/2
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
    df_melt = pd.melt(df3, value_name='segments_back', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\segments_back_{filename}.csv')

    return
def plot_n_fex_km_seg_back(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['f_ex'].isin(list(f_ex_include))]
    print(df3)
    df4 = df3[df3['km_minus'].isin(list(km_include))]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    #
    for i in f_ex_include:

        df5 = df4[df4['f_ex'].isin([i])]
        plt.figure()
        sns.displot(df5, col='team_size', row='km_minus', stat='count', common_norm=False, commen_bin=False).set_title(f'External force = {i}pN {titlestring}')
        #plt.title(f'Distribution (interpolated) of cargo location {titlestring} ')
        plt.xlabel('Cargo back shifts [nm]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_segments_back_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

#>> box(median) or poin/bar(mean)??

## motors ##

'''Motor forces pdf'''
def motorforces_n_fex_km(dirct, ts_list, fex_list, km_list, stepsize=0.1, samplesize=100, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    stepsize : float, default=0.1
               Step size for interpolation function
    samplesize : int, default=100
                Size of the random sample size per iteration (1000 per motor is default) per motor
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

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
            # Unpickle motor0 object
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
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                        mf_sampled = random.sample(list(mf_intrpl), samplesize)
                        # add nested list
                        motor_interpolated_forces.extend(mf_sampled)

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
    df_melt = pd.melt(df, value_name='motor_forces', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)

    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\motorforces_N_fex_km_{filename}.csv')

    return
def plot_fex_N_km_forces_motors(dirct, filename, n_include, fex_include, km_include, stat='probability', show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            for k in km_include:
                df_km = df_n[round(df_n['km'], 2) == round(k, 2)]
                print(df_km)
                #
                plt.figure()
                sns.displot(df_km, x='motor_forces', stat=stat, palette='bright')
                plt.xlabel('Motor forces [pN]')
                plt.title(f'F_ex={i}pN, N={j} motors, km={k}pN/nm')
                plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_fmotors_{i}fex{j}N{k}km_{figname}.png', format='png', dpi=300, bbox_inches='tight')
                if show == True:
                    plt.show()
                    plt.clf()
                    plt.close()
                else:
                    plt.clf()
                    plt.close()
                    print('Figure saved')

    return
# REPORT #DONE
def plot_fex_N_km_forces_motors_cdf(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            plt.figure()
            g = sns.ecdfplot(data=df_n, x="motor_forces", hue="km",  palette='bright')
            #g.legend(title=r'k_{m} [pN/nm]')
            plt.title(f'{j} motors,' r'$F_{ex}= $' f'{i}pN')
            plt.xlabel('Motor forces [pN]')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\ecdf_mf_{i}fex{j}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\ecdf_mf_{i}fex{j}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')
    return
def plot_fex_N_km_forces_motors_violin(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            plt.figure()
            g = sns.violinplot(data=df_n, x="km", y="motor_forces")
            plt.xlabel(r'$k_{m}$ [pN/nm]')
            plt.ylabel('Motor forces [pN]')
            plt.ylim(-10,10)
            plt.yticks(np.arange(-14, 14, 2))
            plt.title(f'{j} motors,' r'$F_{ex}= $' f'{i}pN')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\violin_fm_{i}fex{j}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\violin_fm_{i}fex{j}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return

'''Motor displacement pdf'''
def xm_n_fex_km(dirct, ts_list, fex_list, km_list, stepsize=0.01, samplesize=100, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    stepsize : float, default=0.01
               Step size for interpolation function
    samplesize : int, default=100
                Size of the random sample size per iteration (1000 per motor is default) per motor
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

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
            fex = motor0.f_ex
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
                    print(f'len xm should be 1000: {len(xm)}')
                    #
                    print('Start interpolating distances...')
                    for i, value in enumerate(time):
                        # time points of run i
                        t = value
                        # locations of motors
                        xm_i = xm[i]
                        print(f'fex={fex}')
                        print(xm_i[0])
                        if fex != 0:
                            print('pop')
                            xm_i.pop(0) #mistake I made!!!
                        if fex == 0:
                            print('insert')
                            xm_i.insert(0, 0)
                        print(xm_i[0])
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
                        if len(t) > len(xm_i):
                            t.pop()
                        # Create function
                        print(len(t))
                        print(len(xm_i))
                        f = interp1d(t, xm_i, kind='previous')
                        # New x values, 100 seconds every second
                        interval = (0, t[-1])
                        print(f'interval time: {interval}')
                        t_intrpl = np.arange(interval[0], interval[1], stepsize)
                        # Do interpolation on new data points
                        xm_intrpl = f(t_intrpl)
                        # Random sampling
                        xm_sampled = random.sample(list(xm_intrpl), samplesize)
                        # add nested list
                        xm_interpolated.extend(xm_sampled)

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
    df = pd.DataFrame(nested_xm, index=multi_column).T
    print(df)
    del nested_xm

    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='xm', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\xm_N_fex_km_{filename}.csv')

    return
def plot_fex_N_km_xm(dirct, filename, n_include, fex_include, km_include, stat='probability', show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in fex_include:
        print(i)
        df_fex = df4[df4['f_ex'] == i]
        print(df_fex)
        for j in n_include:
            df_n = df_fex[df_fex['team_size'] == j]
            print(df_n)
            for k in km_include:
                df_km = df_n[round(df_n['km'], 2) == round(k, 2)]
                print(df_km)
                #
                q25, q75 = np.percentile(list(df_km['xm']), [25, 75])
                print(q25)
                print(q75)
                bin_width = 2 * (q75 - q25) * len(list(df_km['xm'])) ** (-1/3)
                print(bin_width)
                if bin_width > 0:
                    max = df_km['xm'].max()
                    print(f'max={max}')
                    min = df_km['xm'].min()
                    print(f'min={min}')
                    bins_xm = round((float(df_km['xm'].max()) - float(df_km['xm'].min())) / float(bin_width))
                else:
                    bins_xm = round(len(list(df_km['xm']))/100)
                plt.figure()
                sns.displot(df2, x='xm', stat=stat, palette='bright', bins=bins_xm)
                plt.xlabel('xm [nm]')
                plt.title(f'F_ex={i}pN, N={j} motors, km={k}pN/nm')
                plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_xm_{i}fex{j}N{k}km_{figname}.png', format='png', dpi=300, bbox_inches='tight')
                if show == True:
                    plt.show()
                    plt.clf()
                    plt.close()
                else:
                    plt.clf()
                    plt.close()
                    print('Figure saved')

    return
def plot_fex_N_km_xm_violin(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:
        print(i)
        df_n = df4[df4['team_size'] == i]
        print(df_n)
        sns.violinplot(data=df_n, x="f_ex", y="xm", hue="km")
        plt.xlabel('External force [pN]')
        plt.ylabel('Motor displacement [nm]')
        plt.title(f'N={i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\violin_xm_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')


    return

'''Motor RL box'''
def rl_motors_n_fex_km(dirct, ts_list, fex_list, km_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    fex_list : list of floats,
                  Values of the external force f_ex within dirct
    km_list : list of floats,
                  Values of the motor stiffness km within dirct
    stepsize : float, default=0.01
               Step size for interpolation function
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_rl = []
    key_tuples = []
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
            key_tuples.append(key)
            #
            rl_all_motors = []
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
                    rl_all_motors.extend(motor.run_length)
            #
            nested_rl.append(rl_all_motors)
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
    print(f'len(nested_rl) should be {len(key_tuples)}: {len(nested_rl)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'f_ex', 'km'])
    print(multi_column)
    del key_tuples

    df = pd.DataFrame(nested_rl, index=multi_column).T
    print(df)
    del nested_rl
    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='rl_motors', var_name=['team_size', 'f_ex', 'km']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}rlmotors_N_fex_km_{filename}.csv')

    return
def plot_fex_N_km_rl_motors(dirct, filename, n_include, fex_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings
                  Which values of the motor_team size N to include
    fex_include : tuple of strings
                  Which values of the external force f_ex to include
    km_include : tuple of strings
                  Which values of the  motor stiffness km to include
    stat :
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['f_ex'].isin(fex_include)]
    print(df3)
    df4 = df3[df3['km'].isin(km_include)]
    print(df4)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:
        df_n = df4[df4['team_size'] == i]
        plt.figure()
        sns.catplot(data=df_n, x='f_ex', y='rl_motors', hue='km', kind='box')
        plt.xlabel('k [pN/nm]')
        plt.ylabel('Motor run length [nm]')
        plt.title(f' {i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_rlmotor_fexNkm_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    for i in fex_include:
        df_n = df4[df4['f_ex'] == i]
        plt.figure()
        sns.catplot(data=df_n, x='f_ex', y='rl_motors', hue='km', kind='box')
        plt.xlabel('k [pN/nm]')
        plt.ylabel('Motor run length [nm]')
        plt.title(f'F_ex {i}pN')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_rlmotor_fexNkm_{i}fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
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



##########################################

##### SYMMETRY, N + KM_RATIO #####
# trap Fu may also be interesting: how close to N x Fs says something about how well the motors with a certain Km work together, how wel do they share load
# RL cargo
# vel cargo


##########################################

##### SYMMETRY, N + KM_RATIO #####
'''Cargo RL ''' # REPORT #DONE
def rl_cargo_n_kmr(dirct, ts_list, kmminus_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : :obj:`list` of :obj:`str`
                  Values of the motor_team size N within dirct
    kmminus_list : list of float,
                  Values of the minus motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None
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
            runlength = list(motor0.runlength_bead) #runlength_cargo
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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\rl_cargo_Nkmr_{filename}.csv', index=False)

    return
def plot_n_kmratio_rl(dirct, filename, n_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subsubdirectory within the 'motor_object' subdirectory within the current working directory
    filename : string
            Name of the dataframe file
    n_include : tuple of strings,  default = ('[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]')
                  Which values of the motor_team size N to include
    km_include : tuple of strings,  default = ('0.1', '0.12', '0.14', '0.16', '0.18', '0.2')
                  Which values of the minus motor stiffness km to include
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed
    figname : string, optional
            Addition to include into the default figure name
    Returns
    -------
    None

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    plt.figure()
    g = sns.catplot(data=df3, x="km_minus", y="run_length", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se')
    g._legend.set_title(r'Team size [$N_{minus}$,$N_{plus}$]')
    plt.xlabel(r'$k_{m}$ minus motors [pN/nm]')
    plt.ylabel('Cargo Runlength [nm]')
    plt.ylim(-2000,3000)
    plt.yticks(np.arange(-2000, 3000, step=500))
    print(f'Start saving...')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_rl_{figname}.png', format='png', dpi=300)
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pp_cargo_rl_{figname}', format='svg', dpi=300)
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return

'''Bind time cargo >> combine with RL? >>> or velocity?? ''' # REPORT #DONE
def bt_cargo_n_kmr(dirct, ts_list, kmminus_list, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    kmminus_list : list of floats,
                  Values of the minus motor stiffness km within dirct
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None
    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    dict_bt = {}
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
            time_bind = list(motor0.time_unbind)
            rl = list(motor0.runlength_bead)
            #
            flat_tb = []
            for i, sub_tb in enumerate(time_bind):
                #print(sub_tb[0])
                #print(len(sub_tb))
                sub_tb.insert(0,0)
                #print(len(sub_tb))
                #print(sub_tb[0])
                diff_tb = np.diff(np.asarray(sub_tb))
                if len(diff_tb) != len(rl[i]):
                    print(f'mistake')
                #print(diff_tb)
                flat_tb.extend(diff_tb)
            #
            dict_bt[key] = flat_tb
            #
            if kmminus_count < len(kmminus_list) - 1:
                kmminus_count += 1
            elif kmminus_count == len(kmminus_list) - 1:
                kmminus_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_bt.items() ]))
    print(df)
    df_melt = pd.melt(df, value_name='time_unbind', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\bt_Nkmr_{filename}.csv', index=False)

    return
def plot_n_kmratio_bt(dirct, filename, n_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    plt.figure()
    g = sns.catplot(data=df3, x="km_minus", y="time_unbind", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se')
    g._legend.set_title(r'Team size [$N_{minus}$,$N_{plus}$]')
    plt.xlabel(r'$k_{m}$ minus motors [pN/nm]')
    plt.ylabel('Cargo Runtime [s]')
    plt.ylim(0,45)
    plt.yticks(np.arange(0, 45, step=5))
    print(f'Start saving...')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_bt_{figname}.png', format='png', dpi=300)
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\pp_cargo_bt_{figname}', format='svg', dpi=300)
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
def xb_n_kmr_2(dirct, ts_list, kmminus_list, stepsize=0.1, samplesize=100, filename=''):
    """
    DONEE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    kmminus_list : list of floats,
                  Values of the minus motor stiffness km within dirct
    stepsize : float, default=0.1
               Step size for interpolation function
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None
    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_xb = []
    key_tuples = []
    #
    teamsize_count = 0
    km_minus_count = 0
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
            print(f'km_minus_count={km_minus_count}')
            # Unpickle motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[km_minus_count]
            print(f'ts={ts}')
            print(f'km_minus={km_minus}')
            #
            key = (str(ts), str(km_minus))
            print(f'key={key}')
            key_tuples.append(key)
            #
            xb = motor0.x_bead #x_cargo
            time = motor0.time_points
            del motor0
            xb_total_interpl = []
            print('Start interpolating bead locations...')
            for index, list_xb in enumerate(xb):
                    #print(f'index={index}')
                    # Original data
                    t_i = time[index]
                    xb_i = list_xb
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
                    if len(t_i) != len(xb_i):
                        t_i.pop()
                    # Create function
                    f = interp1d(t_i, xb_i, kind='previous')
                    # New x values, 100 seconds every second
                    interval = (0, t_i[-1])
                    t_intrpl = np.arange(interval[0], interval[1], stepsize)
                    # Do interpolation on new data points
                    xb_intrpl = f(t_intrpl)
                    # Random sampling
                    xb_sampled = random.sample(list(xb_intrpl), samplesize)
                    # Add to list
                    xb_total_interpl.extend(xb_sampled)

            #
            nested_xb.append(tuple(xb_total_interpl))

            #
            if km_minus_count < len(kmminus_list) - 1:
                km_minus_count += 1
            elif km_minus_count == len(kmminus_list) - 1:
                km_minus_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_nested_xb) should be {len(key_tuples)}: {len(nested_xb)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus'])
    print(multi_column)
    del key_tuples
    #
    df1 = pd.DataFrame(nested_xb, index=multi_column).T
    print(df1)
    del nested_xb
    #
    print('Melt dataframe... ')
    df_melt1 = pd.melt(df1, value_name='xb', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt1)
    del df1
    #
    print('Save dataframe... ')
    df_melt1.to_csv(f'.\motor_objects\\{dirct}\\data\\xb_Nkmr_{filename}.csv', index=False)

    return
def plot_N_kmr_xb_2(dirct, filename, n_include, km_include, stat='probability', show=True, figname=''):
    """
    DONEE
    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        for j in km_include:
            print(j)
            df_km = df_n[round(df_n['km_minus'], 2) == round(j, 2)]
            print(df_km)
            #
            q25, q75 = np.percentile(list(df_km['xb']), [25, 75])
            print(q25)
            print(q75)
            bin_width = 2 * (q75 - q25) * len(list(df_km['xb'])) ** (-1/3)
            print(bin_width)
            if bin_width > 0:
                max = df_km['xb'].max()
                print(f'max={max}')
                min = df_km['xb'].min()
                print(f'min={min}')
                bins_xb = round((float(df_km['xb'].max()) - float(df_km['xb'].min())) / float(bin_width))
            else:
                bins_xb = round(len(list(df_km['xb']))/100)
            plt.figure()
            sns.displot(df_km, x='xb', stat=stat, bins=bins_xb)
            plt.title(f'N= {i} motors, km minus= {j}pN/nm')
            plt.xlabel('xb [nm]')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_xb_{i}N{j}kminus_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return
#REPORT #DONE
def plot_N_kmr_xb_2_cdf(dirct, filename, n_include, km_include, show=True, figname=''):
    """
    DONEE
    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        plt.figure()
        g = sns.ecdfplot(data=df_n, x="xb", hue="km_minus",  palette='bright')
        #g._legend.set_title(r'k_{m} minus motors [pN/nm]')
        plt.xlim(-15000, 15000)
        plt.title(f'{i} motors')
        plt.xlabel('Cargo displacement [nm]')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\ecdf_xb_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\ecdf_xb_{i}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    n = '[4, 4]'
    df_n = df3[df3['team_size'] == n]
    print(df_n)
    plt.figure()
    g = sns.ecdfplot(data=df_n, x="xb", hue="km_minus",  palette='bright')
    plt.xlim(-10000, 0)
    plt.ylim(0, 0.5)
    plt.title(f'{n} motors')
    plt.xlabel('Cargo displacement [nm]')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\ecdf_xb_{n}N_zoommin{figname}.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\ecdf_xb_{n}N_zoommin{figname}', format='svg', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')
    df_n = df3[df3['team_size'] == '[4, 4]']
    print(df_n)
    plt.figure()
    g = sns.ecdfplot(data=df_n, x="xb", hue="km_minus",  palette='bright')
    plt.xlim(0, 10000)
    plt.ylim(0.5, 1)
    plt.title(f'{i} motors')
    plt.xlabel('Cargo displacement [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\ecdf_xb_{i}N_zoomplus{figname}.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\ecdf_xb_{i}N_zoomplus{figname}', format='svg', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return

#not finished
def xb_n_kmr(dirct, ts_list, kmminus_list, stepsize=0.1, filename=''):
    """
    DONE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    kmminus_list : list of floats,
                  Values of the minus motor stiffness km within dirct
    stepsize : float, default=0.1
               Step size for interpolation function
    filename : string, optional
            Addition to include into the default file name

    Returns
    -------
    None
    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_xb = []
    key_tuples = []
    #
    teamsize_count = 0
    km_minus_count = 0
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
            print(f'km_minus_count={km_minus_count}')
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[km_minus_count]
            print(f'ts={ts}')
            print(f'km_minus={km_minus}')
            #
            key = (str(ts), str(km_minus))
            print(f'key={key}')
            key_tuples.append(key)
            #
            xb = motor0.x_cargo
            time = motor0.time_points
            del motor0
            xb_total_interpl = []
            print('Start interpolating bead locations...')
            for index, list_xb in enumerate(xb):
                    #print(f'index={index}')
                    # Original data
                    t_i = time[index]
                    xb_i = list_xb
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
            if km_minus_count < len(kmminus_list) - 1:
                km_minus_count += 1
            elif km_minus_count == len(kmminus_list) - 1:
                km_minus_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_xb)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus'])
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
    df_melt1 = pd.melt(df1, value_name='xb', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt1)
    del df1
    df_melt2 = pd.melt(df2, value_name='xb', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt2)
    del df2
    #
    print('Save dataframe... ')
    df_melt1.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_xb1.csv')
    df_melt2.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_xb2.csv')

    return
def plot_N_kmr_xb(dirct, filename, n_include, km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    if total == False:
        df1 = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename1}')
        df2 = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename2}')
        df_total = pd.concat([df1, df2], ignore_index=True)
        df_total.to_csv(f'.\motor_objects\\{dirct}\\data\\N_kmratio_xb_total.csv')
    else:
        df_total = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\N_kmratio_xb_total.csv')

    df2 = df_total[df_total['team_size'].isin(n_include)]
    print(df2)
    df3 = df2[df2['km_minus'].isin(km_include)]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    plt.figure()
    sns.displot(df3, x='xb',  col='km_ratio', row='team_size', stat=stat)
    plt.title(f'Distribution of cargo location {titlestring} ')
    plt.xlabel('Distance traveled [nm]')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\figures\dist_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return

'''Cargo trajectory examples > + motor trajectories ??''' #REPORT
def traj_n_kmr(dirct, ts_list, kmminus_list, show=True):
    """
    DONE
    Parameters
    ----------
    dirct : string
            Name of the subdirectory within the 'motor_object'
    ts_list : list of lists,
                  Values of the motor_team size N within dirct
    kmminus_list : list of floats,
                  Values of the minus motor stiffness km within dirct
    show : boolean,  default = True
                  If True, display all figures and blocks until the figures have been closed

    Returns
    -------
    None
    """

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
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
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            random_index = random.sample(range(0, 999), 5)
            for i in random_index:
                motor0.time_points[i].pop()
                x = motor0.time_points[i]
                y = motor0.x_bead[i] #x_cargo
                # Plotting
                if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
                    os.makedirs(f'.\motor_objects\\{dirct}\\figures')
                sns.color_palette()
                sns.set_style("whitegrid")
                print('Making figure..')
                plt.step(x, y, where='post')
                #plt.scatter(x,y)
                plt.title(f'Example of trajectory: N={ts}, Km minus={km_ratio}')
                plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_{ts}_{km_ratio}_{i}.png', format='png', dpi=300, bbox_inches='tight')
                if show == True:
                    plt.show()
                    plt.clf()
                    plt.close()
                else:
                    plt.clf()
                    plt.close()
                    print('Figure saved')

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    return

'''bound motors barplot'''
def boundmotors_n_kmr(dirct, ts_list, kmminus_list, stepsize=0.1, filename=''):
    """
    DONEE
    Parameters
    ----------


    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_bound = []
    keys_bound = []
    #
    teamsize_count = 0
    km_minus_count = 0
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
            print(f'km_minus_count={km_minus_count}')
            # Unpickle test_motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[km_minus_count]
            print(f'ts={ts}')
            print(f'km_minus={km_minus}')
            ### antero ###
            key_antero = (str(ts), str(km_minus), 'antero')
            print(f'key_antero={key_antero}')
            keys_bound.append(key_antero)
            #
            antero_bound = motor0.antero_motors # antero_bound
            mean_antero_bound = []
            #
            print('Start interpolating antero bound motors...')
            for index, list_bm in enumerate(antero_bound):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    bound = list_bm
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
            #
            nested_bound.append(mean_antero_bound)
            ### retro ###
            key_retro = (str(ts), str(km_minus), 'retro')
            print(f'key_retro={key_retro}')
            keys_bound.append(key_retro)
            #
            retro_bound = motor0.retro_motors #retro_bound
            mean_retro_bound = []
            #
            print('Start interpolating retro bound motors...')
            for index, list_bm in enumerate(retro_bound):
                    #print(f'index={index}')
                    # Original data
                    t = motor0.time_points[index]
                    bound = list_bm
                    # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
            nested_bound.append(mean_retro_bound)

            #
            if km_minus_count < len(kmminus_list) - 1:
                km_minus_count += 1
            elif km_minus_count == len(kmminus_list) - 1:
                km_minus_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_nested_xb) should be {len(nested_bound)}: {len(nested_bound)}')
    #
    multi_column_antero = pd.MultiIndex.from_tuples(keys_bound, names=['team_size', 'km_minus', 'direction'])
    print(multi_column_antero)
    del keys_bound
    #
    df = pd.DataFrame(nested_bound, index=multi_column_antero).T
    print(df)
    del nested_bound
    #
    print('Melt antero dataframe... ')
    melt_df = pd.melt(df, value_name='motors_bound', var_name=['team_size', 'km_minus', 'direction']).dropna()
    print(melt_df)
    del df

    melt_df.to_csv(f'.\motor_objects\\{dirct}\\data\\anteroretrobound_Nkmratio_{filename}.csv', index=False)

    return
def plot_n_kmr_boundmotors(dirct, filename, n_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Return
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:

        df_n = df3[df3['team_size'] == i]
        print('Making figure..')
        plt.figure()
        sns.barplot(data=df_n, x="km_minus", y="motors_bound", hue="direction", ci=95)
        plt.xlabel('k [pN/nm]')
        plt.ylabel('Bound Motors n')
        plt.title(f' {i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_boundmotors_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
# ADD STATS!!!

'''unbinding events''' #REPORT #DONE
def unbindevent_n_kmr(dirct, ts_list, kmminus_list, filename=''):
    """

    Parameters
    ----------
    DONEE

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')

    #
    nested_unbind = []
    key_tuples = []
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
            key1 = (str(ts), str(km_minus), 'plus')
            print(f'key={key1}')
            key_tuples.append(key1)
            nested_unbind.append(list(motor0.antero_unbinds)) #antero_unbind_events
            #
            key2 = (str(ts), str(km_minus), 'minus')
            print(f'key={key2}')
            key_tuples.append(key2)
            nested_unbind.append(list(motor0.retro_unbinds)) #retro_unbind_events

            #
            if kmminus_count < len(kmminus_list) - 1:
                kmminus_count += 1
            elif kmminus_count == len(kmminus_list) - 1:
                kmminus_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_unbind) should be {len(key_tuples)}: {len(nested_unbind)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus', 'direction'])
    print(multi_column)
    del key_tuples
    #
    df = pd.DataFrame(nested_unbind, index=multi_column).T
    print(df)
    del nested_unbind

    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='unbind_events', var_name=['team_size', 'km_minus', 'direction']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\unbindevents_Nkmr_{filename}.csv', index=False)

    return
def plot_n_kmr_unbindevent(dirct, filename, n_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Return
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:
        df_n = df3[df3['team_size'] == i]
        plt.figure()
        g = sns.barplot(data=df_n, x="km_minus", y="unbind_events", hue="direction", ci=95)
        #g._legend.set_title(r'Team size [N_{minus},N_{plus}]')
        sns.move_legend(g, "upper right")
        plt.xlabel(r'$k_{m}$ minus motors [pN/nm]')
        plt.ylabel('Unbinding events n')
        plt.ylim(0, 1300)
        plt.title(f'{i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_unbindingevents_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\bar_unbindingevents_{i}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
# ADD STATS!!!

'''segments or x vs v segments >> eventueel RL opsplitsen en dan segmenten''' #!
def segment_n_kmr(dirct, ts_list, kmminus_list, filename=''):
    """

    Parameters
    ----------
    DONEE

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
    km_ratio_count = 0
    #
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
            print(f'km_minus_count={km_ratio_count}')
            # Unpickle motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            xb = motor0.x_bead #x_cargo
            t = motor0.time_points
            del motor0
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_minus={km_minus}')
            ##### key tuple #####
            key = (str(ts), str(km_minus))
            print(f'key={key}')
            key_tuples.append(key)
            #
            combined_list = []
            #
            runs_asc, t_asc = st.diff_asc(x_list=xb, t_list=t)
            flat_runs_asc = [element for sublist in runs_asc for element in sublist]
            combined_list.extend(flat_runs_asc)
            #
            runs_desc, t_desc = st.diff_desc(x_list=xb, t_list=t)
            flat_runs_desc = [element for sublist in runs_desc for element in sublist]
            combined_list.extend(flat_runs_desc)
            #
            nested_seg.append(tuple(combined_list))

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_seg)}')
    #
    len_tuples = len(key_tuples)
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus'])
    print(multi_column)
    del key_tuples
    #
    mid_index = len_tuples/2
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
    df_melt = pd.melt(df3, value_name='segments', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\segments_Nkmr_{filename}.csv', index=False)

    return
def plot_n_kmr_seg(dirct, filename, n_include,  km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        for j in km_include:
            print(j)
            df_km = df_n[round(df_n['km_minus'], 2) == round(j, 2)]
            print(df_km)
            #
            q25, q75 = np.percentile(list(df_km['segments']), [25, 75])
            print(q25)
            print(q75)
            bin_width = 2 * (q75 - q25) * len(list(df_km['segments'])) ** (-1/3)
            print(bin_width)
            if bin_width > 0:
                max = df_km['segments'].max()
                print(f'max={max}')
                min = df_km['segments'].min()
                print(f'min={min}')
                bins_xb = round((float(df_km['segments'].max()) - float(df_km['segments'].min())) / float(bin_width))
            else:
                bins_xb = round(len(list(df_km['segments']))/100)
            plt.figure()
            sns.displot(df_km, x='segments', hue='direction', stat=stat, common_norm=True)
            plt.title(f'N= {i} motors, km minus= {j}pN/nm')
            plt.xlabel('Segmented runs [nm]')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_segments_{i}N_{k}kmminus_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return

def seg_back_n_kmr(dirct, ts_list, kmminus_list, filename=''):
    """

    Parameters
    ----------
    DONEE

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
    km_ratio_count = 0
    #
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
            print(f'km_minus_count={km_ratio_count}')
            # Unpickle motor0 object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            xb = motor0.x_cargo
            del motor0
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_minus={km_minus}')
            ##### key tuple #####
            key = (str(ts), str(km_minus))
            print(f'key={key}')
            key_tuples.append(key)
            #
            runs_back = st.diff_unbind(x_list=xb)
            flat_runs_back = [element for sublist in runs_back for element in sublist]
            nested_seg.append(flat_runs_back)
            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_seg)}')
    #
    len_tuples = len(key_tuples)
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus'])
    print(multi_column)
    del key_tuples
    #
    mid_index = len_tuples/2
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
    df_melt = pd.melt(df3, value_name='segments_back', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt)
    #
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\segments_back_{filename}.csv', index=False)

    return
def plot_n_kmr_seg_back(dirct, filename, n_include,  km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        for j in km_include:
            print(j)
            df_km = df_n[round(df_n['km_minus'], 2) == round(j, 2)]
            print(df_km)
            #
            q25, q75 = np.percentile(list(df_km['segments_back']), [25, 75])
            print(q25)
            print(q75)
            bin_width = 2 * (q75 - q25) * len(list(df_km['segments_back'])) ** (-1/3)
            print(bin_width)
            if bin_width > 0:
                max = df_km['segments_back'].max()
                print(f'max={max}')
                min = df_km['segments_back'].min()
                print(f'min={min}')
                bins_xb = round((float(df_km['segments_back'].max()) - float(df_km['segments_back'].min())) / float(bin_width))
            else:
                bins_xb = round(len(list(df_km['segments_back']))/100)
            plt.figure()
            sns.displot(df_km, x='segments_back', stat=stat)
            plt.title(f'N= {i} motors, km minus= {j}pN/nm')
            plt.xlabel('Back movements [nm]')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_segback_{i}N_{k}kmminus_{figname}.png', format='png', dpi=300, bbox_inches='tight')
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
def motorforces_n_kmr_2_sep(dirct, ts_list, kmminus_list, stepsize=0.1, samplesize=25, filename=''):
    """

    Parameters
    ----------
    DONEE
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
            print(os.path.join(path, subdir))
            sub_path = os.path.join(path, subdir)
            #
            print(f'subdir={subdir}')
            print(f'teamsize_count={teamsize_count}')
            print(f'km_ratio_count={km_ratio_count}')
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            antero_interpolated_forces = [] # not nested

            ### loop through motor ANTERO files ###
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file.endswith('anterograde'):
                        #
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
                            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                            mf_sampled = random.sample(list(mf_intrpl), samplesize)
                            # add nested list
                            antero_interpolated_forces.extend(mf_sampled)
                    else:
                        pass
            #
            key = (str(ts), str(km_ratio), 'antero')
            print(f'key={key}')
            key_tuples.append(key)
            nested_motorforces.append(tuple(antero_interpolated_forces))

            #
            retro_interpolated_forces = [] # not nested

            ### loop through motor RETRO files ###
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file.endswith('retrograde'):
                        #
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
                            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                            mf_sampled = random.sample(list(mf_intrpl), samplesize)
                            # add nested list
                            retro_interpolated_forces.extend(mf_sampled)
                    else:
                        pass

            key = (str(ts), str(km_ratio), 'retro')
            print(f'key={key}')
            key_tuples.append(key)
            nested_motorforces.append(tuple(retro_interpolated_forces))

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_motorforces)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus', 'direction'])
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
    df_melt = pd.melt(df, value_name='motor_forces', var_name=['team_size', 'km_minus', 'direction']).dropna()
    print(df_melt)

    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\motorforces_sep_Nkmminus__{filename}.csv', index=False)

    return
#REPORT #DONE
def plot_N_kmr_forces_motors_pdf_sep(dirct, filename, n_include, km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        for j in km_include:
            print(j)
            df_km = df_n[round(df_n['km_minus'], 2) == round(j, 2)]
            print(df_km)
            #
            '''
            q25, q75 = np.percentile(list(df_km['motor_forces']), [25, 75])
            print(q25)
            print(q75)
            bin_width = 2 * (q75 - q25) * len(list(df_km['motor_forces'])) ** (-1/3)
            print(bin_width)
            if bin_width > 0:
                max = df_km['motor_forces'].max()
                print(f'max={max}')
                min = df_km['motor_forces'].min()
                print(f'min={min}')
                bins_mf = round((float(df_km['motor_forces'].max()) - float(df_km['motor_forces'].min())) / float(bin_width))
            else:
                bins_mf = round(len(list(df_km['motor_forces']))/100)
            '''
            plt.figure()
            sns.displot(df_km, x='motor_forces', hue='direction', stat=stat, binwidth=0.5, palette='bright')
            plt.xlabel('Motor forces [pN]')
            plt.ylim(0,0.05)
            plt.yticks(np.arange(0,0.05, 0.005))
            plt.title(f'{i} motors,' r'$k_{m}$ minus=' f'{j} pN/nm')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_fmotors_{i}N_{j}kmminus_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_fmotors_{i}N_{j}kmminus_{figname}', format='svg', dpi=300, bbox_inches='tight')

            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')



    return

#REPORT #DONE
def motorforces_n_kmr_2(dirct, ts_list, kmminus_list, stepsize=0.1, samplesize=50, filename=''):
    """

    Parameters
    ----------
    DONEE
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
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
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
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                        mf_sampled = random.sample(list(mf_intrpl), samplesize)
                        # add nested list
                        motor_interpolated_forces.extend(mf_sampled)

            nested_motorforces.append(tuple(motor_interpolated_forces))
            del motor_interpolated_forces

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')

    #
    print(f'len(nested_motorforces) should be {len(key_tuples)}: {len(nested_motorforces)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus'])
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
    df_melt = pd.melt(df, value_name='motor_forces', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt)

    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\N_kmratio_motorforces_{filename}.csv')

    return
def plot_N_kmr_forces_motors_cdf(dirct, filename, n_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        plt.figure()
        g = sns.ecdfplot(data=df_n, x="motor_forces", hue="km_minus",  palette='bright')
        #g._legend.set_title(r'k_{m} minus motors [pN/nm]')
        plt.xlabel('Motor forces [pN]')
        plt.title(f'{i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\cdf_fmotors_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\cdf_fmotors_{i}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
def plot_N_kmr_forces_motors_violin(dirct, filename, n_include, km_include, show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        plt.figure()
        g = sns.violinplot(data=df_n, x="km_minus", y="motor_forces")
        #g._legend.set_title(r'k_{m} minus motors [pN/nm]')
        plt.xlabel(r'$k_{m}$ minus motors [pN/nm]')
        plt.ylabel('Motor forces [pN]')
        plt.title(f'{i} motors')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\violin_fmotors_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\violin_fmotors_{i}N_{figname}', format='svg', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

#not finished
def motorforces_n_kmr(dirct, ts_list, kmminus_list, stepsize=0.1, filename=''):
    """

    Parameters
    ----------
    DONEE
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
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
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
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
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
    mid_index = 20
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
    #print(df3)

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
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\N_kmratio_motorforces_{filename}.csv')

    return
def plot_N_kmr_forces_motors(dirct, filename, n_include, km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------
    DONE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_ratio'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    #
    for i in n_include:
        df4 = df3[df3['team_size'] == i]
        plt.figure()
        sns.displot(df4, x='motor_forces', col='km_ratio', col_wrap=2, stat='probability', binwidth=0.5, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Motor forces [pN]')
        plt.title(f'Distribution motor forces, motor_team size = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\dist_fmotors_Nkmr_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

'''Motor displacement pdf'''
def xm_n_kmr_2_sep(dirct, ts_list, kmminus_list, stepsize=0.1, samplesize=100, filename=''):
    """

    Parameters
    ----------
    DONEE
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
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')
            #
            antero_xm_interpolated = [] # not nested

            ### loop through motor ANTERO files ###
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file.endswith('anterograde'):
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
                            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                            xm_sampled = random.sample(list(xm_intrpl), samplesize)
                            # add nested list
                            antero_xm_interpolated.extend(xm_sampled)

            key = (str(ts), str(km_ratio), 'antero')
            print(f'key={key}')
            key_tuples.append(key)
            nested_xm.append(tuple(antero_xm_interpolated))

            #
            retro_xm_interpolated = [] # not nested

            ### loop through motor RETRO files ###
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file.endswith('retrograde'):
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
                            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                            xm_sampled = random.sample(list(xm_intrpl), samplesize)
                            # add nested list
                            retro_xm_interpolated.extend(xm_sampled)

            key = (str(ts), str(km_ratio), 'retro')
            print(f'key={key}')
            key_tuples.append(key)
            nested_xm.append(tuple(retro_xm_interpolated))

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    print(f'len(nested_xm) should be {len(key_tuples)}: {len(nested_xm)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus', 'direction'])
    print(multi_column)
    del key_tuples
    #
    df = pd.DataFrame(nested_xm, index=multi_column).T
    print(df)
    del nested_xm

    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='xm', var_name=['team_size', 'km_minus', 'direction']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\xm_sep_Nkmminus_{filename}.csv', index=False)

    return
def plot_N_kmr_xm_sep(dirct, filename, n_include, km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    ### Plotting ###
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    for i in n_include:
        print(i)
        df_n = df3[df3['team_size'] == i]
        print(df_n)
        for j in km_include:
            print(j)
            df_km = df_n[round(df_n['km_minus'], 2) == round(j, 2)]
            print(df_km)
            #

            q25, q75 = np.percentile(list(df_km['xm']), [25, 75])
            print(q25)
            print(q75)
            bin_width = 2 * (q75 - q25) * len(list(df_km['xm'])) ** (-1/3)
            print(bin_width)
            if bin_width > 0:
                max = df_km['xm'].max()
                print(f'max={max}')
                min = df_km['xm'].min()
                print(f'min={min}')
                bins_xm = round((float(df_km['xm'].max()) - float(df_km['xm'].min())) / float(bin_width))
            else:
                bins_xm = round(len(list(df_km['xm']))/100)

            plt.figure()
            sns.displot(df_km, x='xm', hue='direction', stat=stat, palette='bright', bins=bins_xm, common_norm=True)
            plt.xlabel('xm [nm]')
            plt.title(f'N= {i} motors, km minus= {j}pN/nm')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_xm_{i}N_{j}kmminus_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return

#not finished
def xm_n_kmr(dirct, ts_list, kmminus_list, stepsize=0.1, filename=''):
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
            km_ratio = kmminus_list[km_ratio_count]
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
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
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
def xm_n_kmr_2(dirct, ts_list, kmminus_list, stepsize=0.1, samplesize=100, filename=''):
    """

    Parameters
    ----------
    DONE
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
            # Unpickle motor_fixed object
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            time = motor0.time_points
            del motor0
            print(f'len time should be 1000: {len(time)}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
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
                        # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_cargo)
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
                        xm_sampled = random.sample(list(xm_intrpl), samplesize)
                        # add nested list
                        xm_interpolated.extend(xm_sampled)

            nested_xm.append(tuple(xm_interpolated))
            del xm_interpolated

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
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
    df = pd.DataFrame(nested_xm, index=multi_column).T
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
def plot_N_kmr_xm(dirct, filename, n_include, km_include, stat='probability', show=True, figname=''):
    """

    Parameters
    ----------
    DONE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_ratio'].isin(list(km_include))]
    print(df3)
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure...')
    #
    for i in n_include:
        df4 = df3[df3['team_size'] == i]
        plt.figure()
        sns.displot(df4, x='xm', col='km_ratio', col_wrap=2, stat='probability', binwidth=4, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Displacement [nm]')
        plt.title(f' Distribution displacement motors: {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_xm_colN_{i}kmr_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

'''Motor rl'''
def rl_motors_n_kmr_sep(dirct, ts_list, kmminus_list, filename=''):
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
    nested_rl = []
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
            print(f'km_minus_count={km_ratio_count}')
            #
            ts = ts_list[teamsize_count]
            km_ratio = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_ratio={km_ratio}')

            #
            antero_rl = [] # not nested

            ### loop through motor ANTERO files ###
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file.endswith('anterograde'):
                        #
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
                        antero_rl.extend(motor.run_length)
                else:
                    pass

            key = (str(ts), str(km_ratio), 'antero')
            print(f'key={key}')
            key_tuples.append(key)
            nested_rl.append(tuple(antero_rl))

            #
            retro_rl = [] # not nested

            ### loop through motor RETRO files ###
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file.endswith('retrograde'):
                        #
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
                        retro_rl.extend(motor.run_length)
                    else:
                        pass

            key = (str(ts), str(km_ratio), 'retro')
            print(f'key={key}')
            key_tuples.append(key)
            nested_rl.append(tuple(retro_rl))

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    print(f'len(nested_rl) should be {len(key_tuples)}: {len(nested_rl)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus', 'direction'])
    print(multi_column)
    del key_tuples

    df = pd.DataFrame(nested_rl, index=multi_column).T
    print(df)
    del nested_rl
    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='rl_motors', var_name=['team_size', 'km_minus', 'direction']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename}N_kmratio_rl_sep_motors.csv')

    return
def plot_n_kmratio_rl_motors_sep(dirct, filename, n_include, km_include, show=True, figname=''):
    """
    DONE
    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    #
    for i in n_include:
        df4 = df3[df3['team_size'] == i]
        plt.figure()
        sns.displot(df4, x='rl_motors', col='km_minus', col_wrap=3, hue='direction', stat='probability', binwidth=8, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Run length [nm]')
        plt.title(f'Distribution run length motors, motor_team size = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_rlmotors_directionhue_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')


    return

#not finished
def rl_motors_n_kmr(dirct, ts_list, kmminus_list, filename=''):
    """

    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    #
    nested_rl = []
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
            print(f'km_minus_count={km_ratio_count}')
            #
            ts = ts_list[teamsize_count]
            km_minus = kmminus_list[km_ratio_count]
            print(f'ts={ts}')
            print(f'km_minus={km_minus}')
            #
            key = (str(ts), str(km_minus))
            print(f'key={key}')
            key_tuples.append(key)
            #
            rl_all_motors = []
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
                    rl_all_motors.extend(motor.run_length)

            nested_rl.append(rl_all_motors)

            #
            if km_ratio_count < len(kmminus_list) - 1:
                km_ratio_count += 1
            elif km_ratio_count == len(kmminus_list) - 1:
                km_ratio_count = 0
                teamsize_count += 1
            else:
                print('This cannot be right')
    #
    print(f'len(nested_rl) should be {len(key_tuples)}: {len(nested_rl)}')
    #
    multi_column = pd.MultiIndex.from_tuples(key_tuples, names=['team_size', 'km_minus'])
    print(multi_column)
    del key_tuples

    df = pd.DataFrame(nested_rl, index=multi_column).T
    print(df)
    del nested_rl
    '''
    #
    print('Make dataframe from dictionary... ')
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in nested_motorforces.items() ]))
    print(df)
    '''
    print('Melt dataframe... ')
    df_melt = pd.melt(df, value_name='rl_motors', var_name=['team_size', 'km_minus']).dropna()
    print(df_melt)
    #
    print('Save dataframe... ')
    df_melt.to_csv(f'.\motor_objects\\{dirct}\\data\\rlmotors_Nkmminus_{filename}.csv', index=False)

    return
def plot_n_kmratio_rl_motors(dirct, filename, n_include, km_include, show=True, figname=''):
    """
    Parameters
    ----------
    DONEE
    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    print(df)
    df2 = df[df['team_size'].isin(list(n_include))]
    print(df2)
    df3 = df2[df2['km_minus'].isin(list(km_include))]
    print(df3)


    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')

    plt.figure()
    sns.ecdfplot(data=df3, x='team_size', y='rl_motors', hue="km_minus", palette='bright')
    plt.xlabel('k [pN/nm]')
    plt.ylabel('Motor run length [nm]')
    plt.title(f'')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_rlmotors_Nkmminus_{figname}.png', format='png', dpi=300, bbox_inches='tight')
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
    sns.catplot(data=df3, x='km_minus', y='rl_motors', hue='team_size', kind='point')
    plt.xlabel('Trap stiffness of minus motor [pN/nm]')
    plt.ylabel('<Motor run length> [nm]')
    plt.title(f'Run length motors {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\point_rlmotors_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    plt.figure()
    sns.catplot(data=df3, x='km_minus', y='rl_motors', hue='team_size', kind='box')
    plt.xlabel('Run length [nm]')
    plt.title(f'Distribution run length motors {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\box_rlmotors_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    for i in n_include:
        df4 = df3[df3['team_size'] == i]
        plt.figure()
        sns.displot(df4, x='rl_motors', col='km_minus', col_wrap=2, stat='probability', binwidth=4, palette='bright', common_norm=False, common_bins=False)
        plt.xlabel('Run length [nm]')
        plt.title(f'Distribution run length motors, motor_team size = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_rlmotors_colN_{i}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')


    plt.figure()
    sns.catplot(data=df3, x="km_minus", y="run_length", hue="team_size", style='team_size', marker='team_size', kind="point", errornar='se')
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
    '''

    return





''''Quantify asymetry'''
'''(((contourplots??)))'''
'''cdf/pdf(mode), box/violin(median) and lineplot/barplot(mean)'''
'''meanmaxdist #2
def meanmaxdist_n_kmr(dirct, filename, ts_list, kmratio_list, stepsize=1):
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

            # Unpickle motor_fixed object
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
            print(f'motor_team:{motor_team}')
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
                print(f'lists in listoflists should be of equal size: {test}')
                print(f'len(listoflists) should be {length_motorteam}: {len(list_of_lists)}')

                # zip nested list
                print('zip list...')
                zipped = list(zip(*list_of_lists))
                #print(f'print zipped: {zipped}')
                # check zipped list
                test2 = [len(x) for x in zipped]
                print(f'lists of zippedlists should be of equal size, namely {length_motorteam}: unqiue values= {np.unique(np.array(test2))} , type = {type(zipped[0])}')
                print(f'len(zipped) should be same as {test}: {len(zipped)}')
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
'''
