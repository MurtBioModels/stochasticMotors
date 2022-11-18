from scipy.interpolate import interp1d
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def furl(dirct, subdir, figname, titlestring, stat='probability'):

    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')

    # Dictionaries storing data (=values) per number of motors (=key)
    dict_fu = {}
    dict_run_lengths = {}
    # Loop over range of trap stifnesses
    for kt in list_kt:

        # Unpickle motor team
        pickle_file_motorteam = open(f'.\motor_objects\\{dirct}\{subdir}\motorteam', 'rb')
        motorteam = pickle.load(pickle_file_motorteam)
        pickle_file_motorteam.close()

        # Create lists with unbinding forces and run lengths for whole motor team
        fu_list = []
        run_length_list = []
        for motor in motorteam:
            fu_list.extend(motor.forces_unbind)
            run_length_list.extend(motor.run_lengths)
        # Add lists to dictionary at key = number of motor proteins
        dict_fu[kt] = fu_list
        dict_run_lengths[kt] = run_length_list


    # Transform dictionary to Pandas dataframe with equal dimensions
    df_run_length = pd.DataFrame({key:pd.Series(value) for key, value in dict_run_lengths.items()})
    df_force_unbinding = pd.DataFrame({key:pd.Series(value) for key, value in dict_fu.items()})
    # Melt columns
    melted_run_length = pd.melt(df_run_length, value_vars=df_run_length.columns, var_name='Kt').dropna()
    melted_force_unbinding = pd.melt(df_force_unbinding, value_vars=df_force_unbinding.columns, var_name='Kt').dropna()

    # Plot distributions per size of motor team (N motors)
    plt.figure()
    sns.displot(melted_run_length, x='value', hue='Kt', stat=stat, binwidth=8, common_norm=False, palette="bright")
    plt.title(f'Run Length (X) distribution')
    plt.xlabel('Distance [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\\{figname}.png')
    plt.show()

    plt.figure()
    sns.displot(melted_force_unbinding, x='value', hue='Kt', multiple="stack", stat=stat, binwidth=1, common_norm=False, palette="bright")
    plt.title(f'Unbinding force (Fu) distribution')
    plt.xlabel('Unbinding force [pN]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\{figname}.png')
    plt.show()

    # Using FaceGrid
    plt.figure()
    g = sns.FacetGrid(melted_run_length, col="Kt", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, binwidth=8, common_norm=False)
    g.set_axis_labels('Distance [nm]')
    g.set_titles("{col_name}")
    g.fig.suptitle(f'{family}: Run Length (X) distribution per Trap Stiffness (Kt)')
    plt.savefig(f'.\motor_objects\{subject}\{subdir}\\figures\\FGdist_runlength_{calc_eps}eps_{kt}varvalue.png')
    plt.show()

    plt.figure()
    g = sns.FacetGrid(melted_force_unbinding, col="Kt", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, common_norm=False)
    g.set_axis_labels('Unbinding force [pN]')
    g.set_titles("{col_name}")
    g.fig.suptitle(f'{family}: Unbinding force (Fu) distribution per Trap Stiffness (Kt)')
    plt.savefig(f'.\motor_objects\{subject}\{subdir}\\figures\FGdist_fu_{calc_eps}eps_{kt}varvalue.png')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def furl2(dirct, subdir, figname, titlestring, stat='probability'):

    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\{dirct}\{subdir}\\figures')

    # Dictionaries storing data (=values) per number of motors (=key)
    dict_fu = {}
    dict_run_lengths = {}
    # Loop over range of trap stifnesses
    for kt in list_kt:

        # Unpickle motor team
        pickle_file_motorteam = open(f'.\motor_objects\\{dirct}\{subdir}\motorteam', 'rb')
        motorteam = pickle.load(pickle_file_motorteam)
        pickle_file_motorteam.close()

        # Create lists with unbinding forces and run lengths for whole motor team
        fu_list = []
        run_length_list = []
        for motor in motorteam:
            fu_list.extend(motor.forces_unbind)
            run_length_list.extend(motor.run_lengths)
        # Add lists to dictionary at key = number of motor proteins
        dict_fu[kt] = fu_list
        dict_run_lengths[kt] = run_length_list


    # Transform dictionary to Pandas dataframe with equal dimensions
    df_run_length = pd.DataFrame({key:pd.Series(value) for key, value in dict_run_lengths.items()})
    df_force_unbinding = pd.DataFrame({key:pd.Series(value) for key, value in dict_fu.items()})
    # Melt columns
    melted_run_length = pd.melt(df_run_length, value_vars=df_run_length.columns, var_name='Kt').dropna()
    melted_force_unbinding = pd.melt(df_force_unbinding, value_vars=df_force_unbinding.columns, var_name='Kt').dropna()

    # Plot distributions per size of motor team (N motors)
    plt.figure()
    sns.displot(melted_run_length, x='value', hue='Kt', stat=stat, binwidth=8, common_norm=False, palette="bright")
    plt.title(f'Run Length (X) distribution')
    plt.xlabel('Distance [nm]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\\{figname}.png')
    plt.show()

    plt.figure()
    sns.displot(melted_force_unbinding, x='value', hue='Kt', multiple="stack", stat=stat, binwidth=1, common_norm=False, palette="bright")
    plt.title(f'Unbinding force (Fu) distribution')
    plt.xlabel('Unbinding force [pN]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\{figname}.png')
    plt.show()

    # Using FaceGrid
    plt.figure()
    g = sns.FacetGrid(melted_run_length, col="Kt", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, binwidth=8, common_norm=False)
    g.set_axis_labels('Distance [nm]')
    g.set_titles("{col_name}")
    g.fig.suptitle(f'{family}: Run Length (X) distribution per Trap Stiffness (Kt)')
    plt.savefig(f'.\motor_objects\{subject}\{subdir}\\figures\\FGdist_runlength_{calc_eps}eps_{kt}varvalue.png')
    plt.show()

    plt.figure()
    g = sns.FacetGrid(melted_force_unbinding, col="Kt", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, common_norm=False)
    g.set_axis_labels('Unbinding force [pN]')
    g.set_titles("{col_name}")
    g.fig.suptitle(f'{family}: Unbinding force (Fu) distribution per Trap Stiffness (Kt)')
    plt.savefig(f'.\motor_objects\{subject}\{subdir}\\figures\FGdist_fu_{calc_eps}eps_{kt}varvalue.png')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def forces_dist(dirct, subdir, figname, titlestring, stepsize=0.001, stat='probability', cn=False, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\{subdir}\\figures')

    # Unpickle motor team
    pickle_file_motorteam = open(f'.\motor_objects\\{dirct}\{subdir}\motorteam', 'rb')
    motorteam = pickle.load(pickle_file_motorteam)
    pickle_file_motorteam.close()
    # Unpickle test_motor0 object
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    dict_ip_forces = {}
    # Loop through motors in team
    for motor in motorteam:
        forces_ip = []
        # Loop through lists in nested list of bead locations
        for index, list_forces in enumerate(motor.forces):
            print(f'index={index}')

            # Original data
            t = motor0.time_points[index]
            print(len(t))
            forces = list_forces
            print(len(forces))
            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
            if len(t) != len(forces):
                t.pop()
            # Create function
            f = interp1d(t, forces, kind='previous')
            # New x values, 100 seconds every second
            interval = (0, t[-1])
            t_intrpl = np.arange(interval[0], interval[1], stepsize)
            # Do interpolation on new data points
            forces_intrpl = f(t_intrpl)
            # Remove zeroes
            forces_intrpl_nozero = [x for x in forces_intrpl if x != 0]
            if 0 in forces_intrpl_nozero:
                print(f'something went wrong with the zeroes')
            # Add interpolated data points to list of all forces of one motor
            forces_ip.extend(forces_intrpl_nozero)

        # Add list to dictionary with motor family+id as key
        dict_ip_forces[f'{motor.family}_id={motor.id}'] = forces_ip

    # Plotting
    df_forces = pd.DataFrame({key:pd.Series(value) for key, value in dict_ip_forces.items()})
    melted_forces = pd.melt(df_forces, value_vars=df_forces.columns, var_name='id').dropna()

    print('Making figure..')
    plt.figure()
    sns.displot(melted_forces, x='value', hue='id', stat=stat, common_norm=cn, palette="bright")
    plt.title(f'Individual motor forces (interpolated), coloured by indv. motors, common_norm={cn}, {titlestring}')
    plt.xlabel('Force [pN]')
    plt.savefig(f'.\motor_objects\{dirct}\{subdir}\\figures\\dist_motor_forces_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def xm_dist(dirct, subdir, figname, titlestring, stepsize=0.001, stat='probability', cn=False, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\{subdir}\\figures')

    # Unpickle motor team
    pickle_file_motorteam = open(f'.\motor_objects\\{dirct}\\{subdir}\motorteam', 'rb')
    motorteam = pickle.load(pickle_file_motorteam)
    pickle_file_motorteam.close()
    # Unpickle test_motor0 object
    pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    dict_ip_xb = {}
    # Loop through motors in team
    for motor in motorteam:
        xb_ip = []
        # Loop through lists in nested list of bead locations
        for index, list_xm in enumerate(motor.x_m_abs):
            print(f'index={index}')

            # Original data
            t = motor0.time_points[index]
            print(f't={len(t)}')
            print(f'xm={len(list_xm)}')
            # If the last tau draw makes the time overshoot t_end, the Gillespie stops, and t has 1 entry more then force (or x_bead)
            if len(t) != len(list_xm):
                t.pop()
            # Create function
            f = interp1d(t, list_xm, kind='previous')
            # New x values, 100 seconds every second
            interval = (0, t[-1])
            t_intrpl = np.arange(interval[0], interval[1], stepsize)
            # Do interpolation on new data points
            xm_intrpl = f(t_intrpl)
            # Remove zeroes
            xm_intrpl_nozero = [x for x in xm_intrpl if x != 0]
            if 0 in xm_intrpl_nozero:
                print(f'something went wrong with the zeroes')
            # Add interpolated data points to list of all forces of one motor
            xb_ip.extend(xm_intrpl_nozero)

        # Add list to dictionary with motor family+id as key
        dict_ip_xb[f'{motor.family}_id={motor.id}'] = xb_ip

    # Plotting
    df_xb = pd.DataFrame({key:pd.Series(value) for key, value in dict_ip_xb.items()})
    melted_xb = pd.melt(df_xb, value_vars=df_xb.columns, var_name='id').dropna()

    print('Making figure..')
    plt.figure()
    sns.displot(melted_xb, x='value', hue='id', stat=stat, common_norm={cn}, palette="bright")
    plt.title(f'Individual motor distance (interpolated), coloured by indv. motors, common_norm={cn}, {titlestring}')
    plt.xlabel('Distance from x=0 [nM]')
    plt.savefig(f'.\motor_objects\\{dirct}\\{subdir}\\figures\\dist_xm_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def forces_dist_notintrpl(subdir, family, n_motors, kt, calc_eps, subject='varvalue', stat='probability'):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    if not os.path.isdir(f'.\motor_objects\\{subject}\{subdir}\\figures'):
        os.makedirs(f'.\motor_objects\\{subject}\{subdir}\\figures')
    #
    print(f'varvalue={kt}')
    # Unpickle motor team
    pickle_file_motorteam = open(f'.\motor_objects\\{subject}\{subdir}\motorteam', 'rb')
    motorteam = pickle.load(pickle_file_motorteam)
    pickle_file_motorteam.close()
    # Unpickle motor0_Kinesin-1_0.0kt_1motors object
    pickle_file_motor0 = open(f'.\motor_objects\\{subject}\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickle_file_motor0)
    pickle_file_motor0.close()

    dict_ip_forces = {}
    # Loop through motors in team
    for motor in motorteam:
        f_nip = []
    # Loop through lists in nested list of bead locations
        for index, list_forces in enumerate(motor.forces):
            print(f'index={index}')

            f_nip.extend(list_forces)

        # Add list to dictionary with motor family+id as key
        dict_ip_forces[f'{motor.family}_id={motor.id}'] = f_nip

    # Plotting
    df_forces = pd.DataFrame({key:pd.Series(value) for key, value in dict_ip_forces.items()})
    melted_forces = pd.melt(df_forces, value_vars=df_forces.columns, var_name='id').dropna()

    print('Making figure..')
    plt.figure()
    sns.displot(melted_forces, x='value', hue='id', stat=stat, common_norm=True, palette="bright")
    plt.title(f'Individual motor forces (interpolated), coloured by indv. motors, common_norm=True, varvalue={kt}')
    plt.xlabel('Force [pN]')
    plt.savefig(f'.\motor_objects\{subject}\{subdir}\\figures\\NOTintrpl_dist_forces_{calc_eps}eps_{kt}varvalue.png')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


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
