from scipy.interpolate import interp1d
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

### N + FEX + KM >> ELASTIC C. ###
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
    n_select = [1, 2, 3, 4]
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

### N + KM >> ELASTIC C. ###
def plot_N_km_motorforces(dirct, filename, figname=None, titlestring=None, show=False):
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
    # Show the joint distribution using kernel density estimation
    plt.figure()
    g = sns.jointplot(
    data=df.loc[df.km == 0.2],
    x="meanmaxdist", y="runlength", hue="teamsize",
    kind="kde", cmap="bright")
    plt.show()
    '''


    # count plot
    plt.figure()
    g = sns.displot(data=df, x='force', hue='km', col='teamsize', stat='probability', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False, common_bins=False)
    g.fig.suptitle(f' {titlestring}')
    g.set_xlabels(' ')
    g.add_legend()
    plt.show()

    '''
    # hue=teamsize
    plt.figure()
    g = sns.catplot(data=df, x="km", y="meanmaxdist", hue="teamsize", style='teamsize', marker='teamsize', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel(' mean max dist [nm]')
    plt.title(f'{titlestring}')
    plt.show()
    '''
    return


def plot_N_km_xm(dirct, filename, figname=None, titlestring=None, show=False):
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
    # Show the joint distribution using kernel density estimation
    plt.figure()
    g = sns.jointplot(
    data=df.loc[df.km == 0.2],
    x="meanmaxdist", y="runlength", hue="teamsize",
    kind="kde", cmap="bright")
    plt.show()
    '''


    # count plot
    plt.figure()
    g = sns.displot(data=df, x='xm', hue='km', col='teamsize', stat='probability', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False, common_bins=False)
    g.fig.suptitle(f' {titlestring}')
    g.set_xlabels(' ')
    g.add_legend()
    plt.show()

    '''
    # hue=teamsize
    plt.figure()
    g = sns.catplot(data=df, x="km", y="meanmaxdist", hue="teamsize", style='teamsize', marker='teamsize', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel(' mean max dist [nm]')
    plt.title(f'{titlestring}')
    plt.show()
    '''
    return


def plot_N_km_motor_fu(dirct, filename, figname=None, titlestring=None, show=False):
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
    '''
    # Show the joint distribution using kernel density estimation
    plt.figure()
    g = sns.jointplot(
    data=df.loc[df.km == 0.2],
    x="meanmaxdist", y="runlength", hue="teamsize",
    kind="kde", cmap="bright")
    plt.show()
    '''
    plt.figure()
    sns.catplot(data=df[df['km_ratio'].isin(km_select)], x="team_size", y="fu_motors", hue="km_ratio", style='km_ratio', marker='km_ratio', kind="point")

    plt.xlabel('teamsize')
    plt.ylabel('Motor unbinding force [pN]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_fu_motors_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    plt.figure()
    sns.boxplot(data=df[df['km_ratio'].isin(km_select)], x='km_ratio', y='fu_motors', hue='team_size')
    plt.xlabel('k_m ratio')
    plt.ylabel('Motor unbinding force [pN]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\boxplot_fu_motors_{figname}.png', format='png', dpi=300, bbox_inches='tight')
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
    g = sns.displot(data=df, x='fu_motors', hue='km_ratio', col='team_size', stat='probability', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False, common_bins=False)
    g.fig.suptitle(f' {titlestring}')
    g.set_xlabels(' ')
    g.add_legend()
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_fu_motors_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    '''
    # hue=teamsize
    plt.figure()
    g = sns.catplot(data=df, x="km", y="meanmaxdist", hue="teamsize", style='teamsize', marker='teamsize', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel(' mean max dist [nm]')
    plt.title(f'{titlestring}')
    plt.show()
    '''
    return


def plot_N_km_motor_rl(dirct, filename, figname=None, titlestring=None, show=False):
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
    # Show the joint distribution using kernel density estimation
    plt.figure()
    g = sns.jointplot(
    data=df.loc[df.km == 0.2],
    x="meanmaxdist", y="runlength", hue="teamsize",
    kind="kde", cmap="bright")
    plt.show()
    '''


    # count plot
    plt.figure()
    g = sns.displot(data=df, x='rl', hue='km', col='teamsize', stat='probability', multiple="stack",
    palette="bright",
    edgecolor=".3",
    linewidth=.5, common_norm=False, common_bins=False)
    g.fig.suptitle(f' {titlestring}')
    g.set_xlabels(' ')
    g.add_legend()
    plt.show()

    '''
    # hue=teamsize
    plt.figure()
    g = sns.catplot(data=df, x="km", y="meanmaxdist", hue="teamsize", style='teamsize', marker='teamsize', kind="point")
    g._legend.set_title('Team size n=')
    plt.xlabel('Motor stiffness [pN/nm]')
    plt.ylabel(' mean max dist [nm]')
    plt.title(f'{titlestring}')
    plt.show()
    '''
    return



