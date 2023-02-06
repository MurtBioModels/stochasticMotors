from scipy.interpolate import interp1d
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

### N + FEX + KM >> ELASTIC C. ###
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
    print('Making figure..')
    plt.figure()
    sns.displot(df, stat=stat)
    plt.title(f'Distribution (interpolated) of bead location {titlestring} ')
    plt.xlabel('Location [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\figures\dist_xb_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

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
    km_select = [0.02, 0.1, 0.2]
    n_select = [3,4]
    #f_ex = [-4, -5, -6, -7]
    #f_ex = [-0.5, -1, -2, -3, 0]
    #f_ex = [0]
    #
    for i in n_select:

        df2 = df[df['team_size'].isin([i])]
        df3 = df2[df2['km'].isin(km_select)]

        #
        plt.figure()
        sns.catplot(data=df3, x="km", y="run_length", hue="f_ex", style='km', marker='km', kind="point", errornar='se')
        plt.xlabel('km [pN/nm]')
        plt.ylabel('<run length> [nm]')
        plt.title(f' team size N = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_rl_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')

        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')
    '''           
    for i in f_ex:

        df2 = df[df['f_ex'].isin([i])]
        df3 = df2[df2['km'].isin(km_select)]

        plt.figure()
        sns.catplot(data=df3, x='f_ex', y='run_length', hue='km', kind='box')
        plt.xlabel('External force [pN]')
        plt.ylabel('Bead run length [nm]')
        plt.title(f' Teamsize = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_rl_nfexkm_{figname}_{i}N.png', format='png', dpi=300, bbox_inches='tight')
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
        sns.catplot(data=df3, x="team_size", y="run_length", hue="km", style='km', marker='km', kind="point", errornar='se')
        plt.xlabel('teamsize')
        plt.ylabel('<run length> [nm]')
        plt.title(f'f_ex = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_rl_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')

        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')


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

    #
    sns.color_palette()
    sns.set_style("whitegrid")
    km_select = [0.02, 0.1, 0.2]
    fex_select = [-3, -2, -1, -0.5, 0]
    #
    for i in fex_select:
        df2 = df[df['f_ex'].isin([i])]
        df3 = df2[df2['km'].isin(km_select)]
        plt.figure()
        sns.catplot(data=df3, x='team_size', y='bound_motors', hue='km', kind='box')
        plt.xlabel('Team Size N')
        plt.ylabel('Bound Motors n')
        plt.title(f' External force = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_boundmotors_nfexkm_{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')
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
        sns.barplot(data=df3, x="team_size", y="bound_motors", hue="km", ci=95)
        plt.xlabel('Team size N')
        plt.ylabel('<Bound motors>')
        plt.title(f'External force = {i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\barplot_boundmotors_nfexkm{figname}_{i}fex.png', format='png', dpi=300, bbox_inches='tight')

        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return


### N + KM_RATIO >> SYM BREAK ###

def plot_N_kmr_xb(dirct, filename, figname, titlestring, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    #
    df = pd.read_csv(f'.\motor_objects\\{dirct}\\data\\{filename}')
    df_nozeroes = df[df['xb'] !=0 ]
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    # plotting
    sns.set_style("whitegrid")
    km_select = [0.1, 0.5, 1]
    teamsize_select =[str([1,1]), str([2,2]), str([3,3]), str([4,4])]

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

    km_select = [1.0, 0.1, 0.5]
    sns.set_style("whitegrid")

    # hue=km
    plt.figure()
    sns.catplot(data=df, x="km_ratio", y="rl", hue="teamsize", style='km_ratio', marker='km_ratio', kind="point")

    plt.xlabel('km ratio')
    plt.ylabel('<run length> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_rl_kmrxaxis_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    '''
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
    # statistical annotation
    #x1, x2 = 2, 3   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #y, h, col = tips['total_bill'].max() + 2, 2, 'k'
    #plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)
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
    '''

    return

def plot_n_kmratio_boundmotors(dirct, filename, figname=None, titlestring=None, show=False):
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
    ts_select = [[1,1], [2,2], [3,3], [4,4]]

    #
    for i in ts_select:
        df2 = df[df['team_size'].isin([str(i)])]
        plt.figure()
        sns.catplot(data=df2, x='direction', y='motors_bound', hue='km_ratio', kind='box')
        plt.xlabel('direction')
        plt.ylabel('Bound Motors n')
        plt.title(f'km={i} {titlestring}')
        plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_boundmotors_Nkmratio_{figname}_{i}ts.png', format='png', dpi=300, bbox_inches='tight')
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
        sns.barplot(data=df2, x="direction", y="motors_bound", hue="km_ratio", ci=95)
        plt.xlabel('direction')
        plt.ylabel('<Bound motors>')
        plt.title(f'km={i} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\barplot_boundmotors_Nkmratio{figname}_{i}ts.png', format='png', dpi=300, bbox_inches='tight')

        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    return

### N + parameter_RATIO >> SYM BREAK ###

def plot_n_parratio_rl(dirct, filename, parname, figname, titlestring, show=True):
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
    sns.catplot(data=df, x=f"{parname}", y="run_length", hue="team_size", kind="point")

    plt.xlabel(f'{parname}')
    plt.ylabel('<run length> [nm]')
    plt.title(f'{titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\pointplot_rl_parratio_{figname}.png', format='png', dpi=300, bbox_inches='tight')

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    '''
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
    # statistical annotation
    #x1, x2 = 2, 3   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #y, h, col = tips['total_bill'].max() + 2, 2, 'k'
    #plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)
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
    '''
    ts_select = [[1,1], [2,2], [3,3], [4,4]]
    for ts in ts_select:
        # plotting
        plt.figure()
        sns.displot(data=df[df['team_size'].isin([str(ts)])], x='run_length', col=f"{parname}", stat='probability', common_norm=False, palette='bright')
        plt.xlabel('Cargo run length [nm]')
        plt.title(f' N={ts} {titlestring}')
        plt.savefig(f'.\motor_objects\{dirct}\\figures\\dist_rl_parratio_{ts}N_{figname}.png', format='png', dpi=300, bbox_inches='tight')
        if show == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
            print('Figure saved')

    plt.figure()
    sns.catplot(data=df, x=f'team_size', y='run_length', hue=f'{parname}', kind='box')
    plt.xlabel('team size [N-, N+]')
    plt.ylabel('Run length [nm]')
    plt.title(f' {titlestring}')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\box_rl_parratio_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return
