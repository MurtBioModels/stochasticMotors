import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import pickle
import numpy as np
import pandas as pd
import os
from motorgillespie.analysis import segment_trajectories as st
import random
from scipy import stats

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
def plot_n_kmr_unbindevent(dirct, filename, n_include, show=True):
    """

    Parameters
    ----------
    DONEE
    Return
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df2 = df[df['km_minus']==0.25]
    print(df2)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.color_palette()
    sns.set_style("whitegrid")
    print('Making figure..')
    for i in n_include:
        print(f'N={i}')
        df_n = df2[df2['team_size'] == i]
        print(df_n)
        df_pos = df_n[df_n['direction'] == 'plus']
        print(df_pos)
        df_neg = df_n[df_n['direction'] == 'minus']
        print(df_neg)
        stat, p = stats.levene(df_pos['unbind_events'], df_neg['unbind_events'])
        print(f'pvalue levene = {p}')
        shapiro_test_pos = stats.shapiro(df_pos['unbind_events'])
        shapiro_test_neg = stats.shapiro(df_neg['unbind_events'])
        print(f'pvalue shapiro_test_pos = {shapiro_test_pos}')
        print(f'pvalue shapiro_test_neg = {shapiro_test_neg}')

        ttest_false = stats.ttest_ind(df_pos['unbind_events'], df_neg['unbind_events'], equal_var=False)
        ttest_true = stats.ttest_ind(df_pos['unbind_events'], df_neg['unbind_events'], equal_var=True)
        print(f'ttest_false = {ttest_false}')
        print(f'ttest_true = {ttest_true}')
        plt.figure()
        g = sns.histplot(data=df_n, x="unbind_events", hue="direction")
        plt.title(f'{i} motors')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return
