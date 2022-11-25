import pandas as pd
import seaborn as sns
from scipy import stats
import itertools
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import os


def asteriks(pvalues):
    """

    Parameters
    ----------

    Returns
    -------

    """
    asteriks_dict = {}

    for key, pvalue in pvalues.items():
        if pvalue[-1] <= 0.0001:
            asteriks_dict[key] = ["****", pvalue[0], pvalue[1]]
        elif pvalue[-1] <= 0.001:
            asteriks_dict[key] = ["***", pvalue[0], pvalue[1]]
        elif pvalue[-1] <= 0.01:
            asteriks_dict[key] = ["**", pvalue[0], pvalue[1]]
        elif pvalue[-1] <= 0.05:
            asteriks_dict[key] = ["*", pvalue[0], pvalue[1]]
        else:
            asteriks_dict[key] = ["ns", pvalue[0], pvalue[1]]

    return asteriks_dict


def kstest_nkmratio_rl(dirct, data_file, filename_out, team_size, km_ratio):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{data_file}')

    dict_pvalues = {}
    for i in team_size:
        df2 = df[(df["teamsize"] == str(i))]
        print(f'df2={df2}')

        for a, b in itertools.combinations(km_ratio, 2):
            print(a)
            df3 = df2[df2["km_ratio"] == a]
            print(f'df3={df3}')
            sample1 = df3['rl'].to_numpy()
            #print(f'sample1={sample1}')

            df4 = df2[df2["km_ratio"] == b]
            print(f'df4={df4}')
            sample2 = df4['rl'].to_numpy()
            #print(f'sample2={sample2}')
            #
            stat, pvalue = stats.ks_2samp(sample1, sample2)
            #
            key = (str(i), f'{a}vs{b}')
            dict_pvalues[key] = [stat, pvalue]

    for i in km_ratio:
        df2 = df[df["km_ratio"] == i]
        print(f'df2={df2}')

        for a, b in itertools.combinations(team_size, 2):

            df3 = df2[(df2["teamsize"] == str(a))]
            print(f'df3={df3}')
            sample1 = df3['rl'].to_numpy()
            #print(f'sample1={sample1}')

            df4 = df2[(df2["teamsize"] == str(b))]
            print(f'df4={df4}')
            sample2 = df4['rl'].to_numpy()
            #print(f'sample2={sample2}')
            #
            stat, pvalue = stats.ks_2samp(sample1, sample2)
            #
            key = (str(i), f'{a}vs{b}')
            dict_pvalues[key] = [stat, pvalue]

    asteriks_dict = asteriks(dict_pvalues)
    print(asteriks_dict)

    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in asteriks_dict.items() ])).T
    df.columns=['asterik', 'stat', 'pvalue']
    print(df)

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\data'):
        os.makedirs(f'.\motor_objects\\{dirct}\\data')
    df.to_csv(f'.\motor_objects\\{dirct}\\data\\{filename_out}stats_N_kmratio_rl.csv')

    return


def trying(dirct, data_file, team_size, km_ratio):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    km_select = [0.1, 0.5, 1]
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{data_file}')
    df = df[df['km_ratio'].isin(km_select)]
    print(df)
    pvalues = []
    pairs = []
    for i in team_size:
        df2 = df[(df["teamsize"] == str(i))]
        print(f'df2={df2}')

        for a, b in itertools.combinations(km_ratio, 2):
            print(a)
            df3 = df2[df2["km_ratio"] == a]
            print(f'df3={df3}')
            sample1 = df3['rl'].to_numpy()
            #print(f'sample1={sample1}')

            df4 = df2[df2["km_ratio"] == b]
            print(f'df4={df4}')
            sample2 = df4['rl'].to_numpy()
            #print(f'sample2={sample2}')
            #
            stat, pvalue = stats.ks_2samp(sample1, sample2)
            #
            pvalues.append(pvalue)
            pairs.append(  [(str(i), a), (str(i), b) ] )

    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]

    print(pairs)
    print(pvalues)
    #
    sns.set_style("whitegrid")
    plotting_parameters = {
    'data':    df,
    'x':       'teamsize',
    'y':       'rl',
    'hue': 'km_ratio'
}

    # Create new plot
    fig, ax = plt.subplots()

    # Plot with seaborn
    sns.boxplot(**plotting_parameters)

    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.configure(text_format='star', loc='inside')
    annotator.set_pvalues_and_annotate(pvalues)

    # Label and show
    plt.xlabel('teamsize')
    plt.ylabel('Cargo run length [nm]')
    plt.title(f'Boxplot of cargo run length by two teams of motors')

    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_rl_ANNOTATED.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

    return

def trying_hueN(dirct, data_file, team_size, km_ratio):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    km_select = [0.1, 0.5, 1]
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{data_file}')
    df = df[df['km_ratio'].isin(km_select)]
    print(df)
    pvalues = []
    pairs = []
    for i in km_ratio:
        df2 = df[(df["km_ratio"] == i)]
        print(f'df2={df2}')

        for a, b in itertools.combinations(team_size, 2):
            print(a)
            df3 = df2[df2["teamsize"] ==str(a)]
            print(f'df3={df3}')
            sample1 = df3['rl'].to_numpy()
            #print(f'sample1={sample1}')

            df4 = df2[df2["teamsize"] == str(b)]
            print(f'df4={df4}')
            sample2 = df4['rl'].to_numpy()
            #print(f'sample2={sample2}')
            #
            stat, pvalue = stats.ks_2samp(sample1, sample2)
            #
            pvalues.append(pvalue)
            pairs.append(  [(i, str(a)), (i, str(b)) ] )

    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]

    print(pairs)
    print(pvalues)
    #
    sns.set_style("whitegrid")
    plotting_parameters = {
    'data':    df,
    'x':       'teamsize',
    'y':       'rl',
    'hue': 'km_ratio'
}

    # Create new plot
    fig, ax = plt.subplots()

    # Plot with seaborn
    sns.boxplot(**plotting_parameters)

    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.configure(text_format='star', loc='inside')
    annotator.set_pvalues_and_annotate(pvalues)

    # Label and show
    plt.xlabel('teamsize')
    plt.ylabel('Cargo run length [nm]')
    plt.title(f'Boxplot of cargo run length by two teams of motors')

    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_rl_ANNOTATED.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

    return

def ttest(dirct, data_file, team_size, km_ratio):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    km_select = [0.1, 0.5, 1]
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{data_file}')
    df = df[df['km_ratio'].isin(km_select)]
    print(df)
    pvalues = []
    pairs = []
    for i in team_size:
        df2 = df[(df["teamsize"] == str(i))]
        print(f'df2={df2}')

        for a, b in itertools.combinations(km_ratio, 2):
            print(a)
            df3 = df2[df2["km_ratio"] == a]
            print(f'df3={df3}')
            sample1 = df3['rl'].to_numpy()
            #print(f'sample1={sample1}')

            df4 = df2[df2["km_ratio"] == b]
            print(f'df4={df4}')
            sample2 = df4['rl'].to_numpy()
            #print(f'sample2={sample2}')
            #
            stat, pvalue = stats.ttest_ind(sample1, sample2, equal_var=False)
            #
            pvalues.append(pvalue)
            pairs.append(  [(str(i), a), (str(i), b) ] )

    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]

    print(pairs)
    print(pvalues)
    #
    sns.set_style("whitegrid")
    plotting_parameters = {
    'data':    df,
    'x':       'teamsize',
    'y':       'rl',
    'hue': 'km_ratio',
    'kind' : 'point'
}

    # Create new plot
    fig, ax = plt.subplots()

    # Plot with seaborn
    sns.catplot(**plotting_parameters)

    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.configure(text_format='star', loc='inside')
    annotator.set_pvalues_and_annotate(pvalues)

    # Label and show
    plt.xlabel('teamsize')
    plt.ylabel('Cargo run length [nm]')
    plt.title(f'Boxplot of cargo run length by two teams of motors')

    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_rl_ANNOTATED.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

    return


