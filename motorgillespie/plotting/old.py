import pickle
import matplotlib.pyplot as plt
import os.path
import numpy as np
import seaborn as sns
import pandas as pd

def rl_fu_bead2(family, kt, n_motors, n_it, epsilon, stat="probability"):
    """


    Parameters
    ----------

    Returns
    -------

    """
    # Dictionaries storing data (=values) per number of motors (=key)
    dict_run_length_bead = {}
    # Loop over range of motor team sizes
    for n in range(1, n_motors + 1):

        # Unpickle motor_0 object
        pickle_file_motor0 = open(f'Motor0_{family}_{epsilon}_{kt}kt_{n_it}it_{n}motors', 'rb')
        motor0 = pickle.load(pickle_file_motor0)
        pickle_file_motor0.close()
        #
        list_xbead = motor0.x_bead
        flattened_list = [val for sublist in list_xbead for val in sublist]
        size = len(flattened_list)
        index_list = [i + 1 for i, val in enumerate(flattened_list) if val == 0]
        split_list = [flattened_list[start: end] for start, end in
                      zip([0] + index_list, index_list + ([size] if index_list[-1] != size else []))]
        # print(split_list)
        list_rl = []
        for run in split_list:
            list_rl.append(max(run))
        list_rl = [i for i in list_rl if i != 0]
        dict_run_length_bead[n] = list_rl

    # Remove empty keys first
    # print(dict_run_length_bead)
    new_dict = {k: v for k, v in dict_run_length_bead.items() if v}

    # Transform dictionary to Pandas dataframe with equal dimensions
    df_run_lengths = pd.DataFrame({key: pd.Series(value) for key, value in new_dict.items()})
    # Melt columns (drop NaN rows)
    melted_run_lengths = pd.melt(df_run_lengths, value_vars=df_run_lengths.columns, var_name='N_motors').dropna()
    print(melted_run_lengths)

    # Plot distributions per size of motor team (N motors)
    g = sns.FacetGrid(melted_run_lengths, col="N_motors", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, binwidth=1, common_norm=False)
    g.set_axis_labels("Run length of bead [x]")
    g.set_titles("{col_name}")
    g.set(xticks=np.arange(0, 900, 100))
    g.set(xlim=(0, 900))
    plt.savefig(f'DistRLBead_{family}_{epsilon}_{kt}Kt.png')
    plt.show()

    return

def dist_act_motors2(family, kt, n_motors, n_it, epsilon, stat="probability"):
    """


    Parameters
    ----------

    Returns
    -------

    """
    # Dictionaries storing data (=values) per number of motors (=key)
    dict_active_motors = {}
    # Loop over range of motor team sizes
    for n in range(1, n_motors + 1):
        # Unpickle motor_0 object
        pickle_file_motor0 = open(f'Motor0_{family}_{epsilon}_{kt}kt_{n_it}it_{n}motors', 'rb')
        motor0 = pickle.load(pickle_file_motor0)
        pickle_file_motor0.close()

        # Append attribute 'bound motors': list of active motors per time step
        dict_active_motors[n] = motor0.antero_motors

    # Transform dictionary to Pandas dataframe with equal dimensions
    df_bound_motors = pd.DataFrame({key: pd.Series(value) for key, value in dict_active_motors.items()})
    # Melt columns (drop NaN rows)
    melted_bound_motors = pd.melt(df_bound_motors, value_vars=df_bound_motors.columns, var_name='N_motors').dropna()
    print(melted_bound_motors.head())

    # Plot distributions per size of motor team (N motors)
    g = sns.FacetGrid(melted_bound_motors, col="N_motors", col_wrap=5)
    g.map(sns.histplot, "value", stat=stat, discrete=True, common_norm=False)
    g.set_axis_labels("Active motors")
    g.set_titles("{col_name}")
    g.set(xticks=np.arange(0, 11, 1))
    plt.savefig(f'DistActiveMotors_{family}_{epsilon}_{kt}Kt.png')
    plt.show()

    return

##### lineplots beginning ########

# Lineplots of average runlength and unbinding force per varval

def lineplots_kt(family, list_kt, n_motors, n_it, epsilon, file_name=None):
    """

    Parameters
    ----------

    Returns
    -------

    """
    ### Collect data from saved motor objects ###
    # File name varval
    if file_name is None:
        data_file_name = f'SimulationOutput_{family}_{epsilon}Eps_{n_motors}motors'
    else:
        data_file_name = file_name
    # Check if the correct data filename is present in working directory
    if os.path.isfile(data_file_name):
        print("File with this name already present in working directory")
    else:
        # Array holding data
        output_array = np.zeros([len(list_kt), 4])
        # Iteration tracker
        it = 0
        # Loop over list of Trap Stiffness (Kt) values
        for kt in list_kt:
            # Unpickle list of motor objects
            pickle_file_team = open(f'MotorTeam_{kt}kt_{family}_{epsilon}_{n_motors}motors_{n_it}it', 'rb')
            motor_team = pickle.load(pickle_file_team)
            pickle_file_team.close()
            # Unpickle motor_0 object
            pickle_file_motor0 = open(f'Motor0_{kt}kt_{family}_{epsilon}_{n_motors}motors_{n_it}it', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()

            # Create lists with (1) mean unbinding force per motor
            # (2) mean Run Length per motor
            # (3) mean force per motor
            mean_fu_motors = []
            mean_run_motors = []

            for motor in motor_team:
                mean_fu_motors.append((sum(motor.f_tot_unbind) / len(motor.f_tot_unbind)))
                mean_run_motors.append((sum(motor.run_lengths) / len(motor.run_lengths)))

            # Total mean value for whole team
            mean_fu_total = sum(mean_fu_motors) / len(mean_fu_motors)
            mean_run_total = sum(mean_run_motors) / len(mean_run_motors)

            # Add values to array at entry 'it'
            # Average bound motors is added in the 4th column
            output_array[it, 0] = kt
            output_array[it, 1] = mean_fu_total
            output_array[it, 2] = mean_run_total
            output_array[it, 3] = sum(motor0.antero_motors) / len(motor0.antero_motors)

            # Update iteration number
            it += 1

        # Save output array to filename
        np.savetxt(data_file_name, output_array, fmt='%1.3f', delimiter='\t')


    ### Plotting ###
    output_gillespie = np.loadtxt(data_file_name)
    output_analytical = np.loadtxt(f'AnalyticalPerKt_{family}_{len(list_kt)}nKts')

    # Plot mean unbinding Force Gillespie vs Analytical as a function of Trap stiffness (Kt)
    plt.plot(output_gillespie[:,0], output_gillespie[:,1], label="Gillespie <F>", linestyle='--', marker='o', color='b')
    plt.plot(output_analytical[:,0], output_analytical[:,1], label="Analytical <F>", linestyle=':', marker='x', color='r')
    plt.xlabel('Trap stiffness [pN/nm] ')
    plt.ylabel('Mean unbinding force [pN]')
    plt.title(f'{family}: Mean unbinding force <F> as a function of trap stiffness (Kt)')
    plt.legend()
    plt.savefig(f'{family}_{epsilon}_{n_motors}_Fu.png')
    plt.show()
    # Plot mean Run Length Gillespie vs Analytical as a function of Trap stiffness (Kt)
    plt.plot(output_gillespie[:,0], output_gillespie[:,2], label="Gillespie <X>", linestyle='--', marker='o', color='b')
    plt.plot(output_analytical[:,0], output_analytical[:,2], label="Analytical <X>", linestyle=':', marker='x', color='r')
    plt.xlabel('Trap stiffness [pN/nm] ')
    plt.ylabel('Mean Run Length [nm]')
    plt.title(f'{family}: Mean Run Length <X> as a function of trap stiffness (Kt)')
    plt.legend()
    plt.savefig(f'{family}_{epsilon}_{n_motors}_RunLength.png')
    plt.show()

    return

def lineplots_Nmotors(family, kt, n_motors, n_it, epsilon, file_name=None):
    """

    Parameters
    ----------

    Returns
    -------

    """
    ### Collect data from saved motor objects ###
    # File name varval
    if file_name is None:
        data_file_name = f'SimulationOutput_{family}_{epsilon}Eps_{kt}Kt_{n_it}it_{n_motors}motors'
    else:
        data_file_name = file_name
    # Check if the correct data filename is present in working directory
    if os.path.isfile(data_file_name):
        print("File with this name already present in working directory")
    else:
        # Array holding data
        output_array = np.zeros([n_motors, 5])
        # Iteration tracker
        counter = 0
        # Loop over range of motor team sizes
        for n in range(1, n_motors + 1):

            # Unpickle list of motor objects
            pickle_file_team = open(f'MotorTeam_{family}_{epsilon}_{kt}kt_{n_it}it_{n}motors', 'rb')
            motor_team = pickle.load(pickle_file_team)
            pickle_file_team .close()
            # Unpickle motor_0 object
            pickle_file_motor0 = open(f'Motor0_{family}_{epsilon}_{kt}kt_{n_it}it_{n}motors', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()

            # Create lists with (1) mean unbinding force per motor
            # (2) mean Run Length per motor
            mean_fu_motors = []
            mean_run_motors = []
            for motor in motor_team:
                mean_fu_motors.extend(motor.f_tot_unbind)
                mean_run_motors.extend(motor.run_lengths)

            # Total mean value for whole team
            mean_fu_total = (sum(mean_fu_motors) / len(mean_fu_motors))
            mean_run_total = (sum(mean_run_motors) / len(mean_run_motors))
            #
            list_xbead = motor0.x_bead
            flattened_list = [val for sublist in list_xbead for val in sublist]
            size = len(flattened_list)
            index_list = [i + 1 for i, val in enumerate(flattened_list) if val == 0]
            split_list = [flattened_list[start: end] for start, end in zip([0] + index_list, index_list + ([size] if index_list[-1] != size else []))]
            #print(split_list)
            list_rl = []
            for run in split_list:
                list_rl.append(max(run))
            list_rl = [i for i in list_rl if i != 0]

            # Add values to array at entry 'it'
            # Mean bound motors is added in the 4th column and mean bead run length to the 5th
            output_array[counter, 0] = n
            output_array[counter, 1] = mean_fu_total
            output_array[counter, 2] = mean_run_total
            output_array[counter, 3] = sum(motor0.antero_motors) / len(motor0.antero_motors)
            if len(list_rl) > 0:
                output_array[counter, 4] = sum(list_rl)/len(list_rl)
            else:
                output_array[counter, 4] = None

            # Update iteration number
            counter += 1

        # Save output array to filename
        np.savetxt(data_file_name, output_array, fmt='%1.3f', delimiter='\t')

    ### Plotting ###
    output_gillespie = np.loadtxt(f'SimulationOutput_{family}_{epsilon}Eps_{kt}Kt_{n_it}it_{n_motors}motors')

    # Plot simulated mean unbinding Force as a function of motor team size (N motors)
    plt.plot(output_gillespie[:,0], output_gillespie[:,1], label="simulation <F>", linestyle='--', marker='o', color='b')
    plt.xlabel('Number of active motor proteins (N)')
    plt.ylabel('Mean unbinding force [pN]')
    plt.title(f'{family}: Mean unbinding force <F> as a function motor team size')
    plt.legend()
    plt.savefig(f'{family}_{epsilon}_{kt}_{n_motors}motors_Fu.png')
    plt.show()

    # Plot simulated mean Run Length as a function of motor team size (N motors)
    plt.plot(output_gillespie[:,0], output_gillespie[:,2], label="simulation <motor run length>", linestyle=':', marker='o', color='r')
    plt.xlabel('Number of active motor proteins (N)')
    plt.ylabel('Mean motor Run Length [nm]')
    plt.title(f'{family}: Mean motor Run Length <X> as a function of motor team size')
    plt.legend()
    plt.savefig(f'{family}_{epsilon}_{kt}_{n_motors}motors_RunLength.png')
    plt.show()

    # Plot simulated mean active motors as a function of motor team size (N motors)
    plt.plot(output_gillespie[:,0], output_gillespie[:,3], label="simulation <bound motors>", linestyle=':', marker='o', color='r')
    plt.xlabel('Number of active motor proteins (N)')
    plt.ylabel('Mean bound motors (n)')
    plt.title(f'{family}: Mean active motors as a function of motor team size')
    plt.legend()
    plt.savefig(f'{family}_{epsilon}_{kt}_{n_motors}motors_BoundMotors.png')
    plt.show()

    # Plot simulated mean ead Run Length as a function of motor team size (N motors)
    plt.plot(output_gillespie[:,0], output_gillespie[:,4], label="simulation <bead run length>", linestyle=':', marker='o', color='r')
    plt.xlabel('Number of active motor proteins (N)')
    plt.ylabel('Mean bead Run Length <X>')
    plt.title(f'{family}: Mean bead Run Length <X>  as a function of motor team size')
    plt.legend()
    plt.savefig(f'{family}_{epsilon}_{kt}_{n_motors}motors_RLBead.png')
    plt.show()

    return

def lineplots_radius(family, k_t, list_r, n_motors, n_it, file_name=None):
    """

    Parameters
    ----------

    Returns
    -------

    """
    ### Collect data from saved motor objects ###
    # File name varval
    if file_name is None:
        data_file_name = f'SimulationOutputR_{family}_{k_t}kt_{n_motors}motors'
    else:
        data_file_name = file_name
    # Check if the correct data filename is present in working directory
    if os.path.isfile(data_file_name):
        print("File with this name already present in working directory")
    else:
        # Array holding data
        output_array = np.zeros([len(list_r), 6])
        # Iteration tracker
        it = 0
        # Loop over list of Trap Stiffness (Kt) values
        for r in list_r:
            # Unpickle list of motor objects
            pickle_file_team = open(f'MotorTeam_{family}_{k_t}kt_{r}radius_{n_motors}motors_{n_it}it', 'rb')
            motor_team = pickle.load(pickle_file_team)
            pickle_file_team.close()
            # Unpickle motor_0 object
            pickle_file_motor0 = open(f'Motor0_{family}_{k_t}kt_{r}radius_{n_motors}motors_{n_it}it', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()

            # Create lists with mean values per motor
            mean_fx = []
            mean_fz = []
            mean_run_motors = []

            for motor in motor_team:
                mean_fx.append((sum(motor.fx_unbind) / len(motor.fx_unbind)))
                mean_fz.append((sum(motor.fz_unbind) / len(motor.fz_unbind)))
                mean_run_motors.append((sum(motor.run_lengths) / len(motor.run_lengths)))

            # Total mean values for whole team
            mean_fx_total = sum(mean_fx) / len(mean_fx)
            mean_fz_total = sum(mean_fz) / len(mean_fz)
            mean_run_total = sum(mean_run_motors) / len(mean_run_motors)

            # Add values to array at entry 'it'
            output_array[it, 0] = r
            output_array[it, 1] = motor0.angle
            output_array[it, 2] = mean_fx_total
            output_array[it, 3] = mean_fz_total
            output_array[it, 4] = mean_run_total

            # Update iteration number
            it += 1

        # Save output array to filename
        np.savetxt(data_file_name, output_array, fmt='%1.3f', delimiter='\t')

    ### Plotting ###

    output_gillespie = np.loadtxt(data_file_name)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(output_gillespie[:,0], output_gillespie[:,2], label="<Fx>", linestyle='-', marker='+', color='r')
    ax1.plot(output_gillespie[:,0], output_gillespie[:,3], label="<Fz>", linestyle='--', marker='o', color='y')
    ax2.plot(output_gillespie[:,0], output_gillespie[:,4], label="<X>", linestyle='-.', marker='^', color='g')

    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], len(list_r)))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], len(list_r)))

    ax1.set_xlabel('Radius bead [nm]')
    ax1.set_ylabel('Mean unbinding force [pN]')
    #ax2.set_xlabel('Angle [degrees]')
    ax2.set_ylabel('Mean Run Length [nm]')

    fig.suptitle(f'{family}: Mean unbinding force <F> and run length <X> as a function of bead radius')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()
    plt.savefig(f'Radius_{family}_{k_t}kt_{n_motors}motors_{n_it}it.png')
    plt.show()

    '''
    output_gillespie = np.loadtxt(data_file_name)
    plt.plot(output_gillespie[:,0], output_gillespie[:,2], label="<Fx>", linestyle='-', marker='+', color='r')
    plt.plot(output_gillespie[:,0], output_gillespie[:,3], label="<Fz>", linestyle='--', marker='o', color='y')
    plt.ylabel('Mean unbinding force [pN]')
    plt.yscale('log')
    plt.xlabel('Radius bead [nm]')
    plt.title(f'{family}: Mean unbinding force <F> and run length <X> as a function of bead radius')
    plt.legend()
    plt.savefig(f'logscale_force_radius_{family}_{k_t}kt_{n_motors}motors_{n_it}it.png')
    plt.show()

    plt.plot(output_gillespie[:,0], output_gillespie[:,4], label="<X>", linestyle='-.', marker='^', color='g')
    plt.ylabel('Mean unbinding force [pN]')
    plt.yscale('log')
    plt.xlabel('Mean Run Length [nm]')
    plt.title(f'{family}: Mean unbinding force <F> and run length <X> as a function of bead radius')
    plt.legend()
    plt.savefig(f'logscale_RL_radius_{family}_{k_t}kt_{n_motors}motors_{n_it}it.png')
    plt.show()
    '''
    return

def lineplots_rl(family, k_t, radius, list_rl, n_motors, n_it, file_name=None):
    """

    Parameters
    ----------

    Returns
    -------

    """
    ### Collect data from saved motor objects ###
    # File name varval
    if file_name is None:
        data_file_name = f'SimulationOutputRL_{radius}r_{family}_{k_t}kt_{n_motors}motors'
    else:
        data_file_name = file_name
    # Check if the correct data filename is present in working directory
    if os.path.isfile(data_file_name):
        print("File with this name already present in working directory")
    else:
        # Array holding data
        output_array = np.zeros([len(list_rl), 6])
        # Iteration tracker
        it = 0
        # Loop over list of Trap Stiffness (Kt) values
        for rl in list_rl:
            # Unpickle list of motor objects
            pickle_file_team = open(f'MotorTeam_{family}_{k_t}kt_{radius}r_{rl}rl_{n_motors}motors_{n_it}it', 'rb')
            motor_team = pickle.load(pickle_file_team)
            pickle_file_team.close()
            # Unpickle motor_0 object
            pickle_file_motor0 = open(f'Motor0_{family}_{k_t}kt_{radius}r_{rl}rl_{n_motors}motors_{n_it}it', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()

            # Create lists with mean values per motor
            mean_fx = []
            mean_fz = []
            mean_run_motors = []

            for motor in motor_team:
                mean_fx.append((sum(motor.fx_unbind) / len(motor.fx_unbind)))
                mean_fz.append((sum(motor.fz_unbind) / len(motor.fz_unbind)))
                mean_run_motors.append((sum(motor.run_lengths) / len(motor.run_lengths)))

            # Total mean values for whole team
            mean_fx_total = sum(mean_fx) / len(mean_fx)
            mean_fz_total = sum(mean_fz) / len(mean_fz)
            mean_run_total = sum(mean_run_motors) / len(mean_run_motors)

            # Add values to array at entry 'it'
            output_array[it, 0] = rl
            output_array[it, 1] = motor0.angle
            output_array[it, 2] = mean_fx_total
            output_array[it, 3] = mean_fz_total
            output_array[it, 4] = mean_run_total

            # Update iteration number
            it += 1

        # Save output array to filename
        np.savetxt(data_file_name, output_array, fmt='%1.3f', delimiter='\t')

    ### Plotting ###

    output_gillespie = np.loadtxt(data_file_name)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(output_gillespie[:,0], output_gillespie[:,2], label="<Fx>", linestyle='-', marker='+', color='r')
    ax1.plot(output_gillespie[:,0], output_gillespie[:,3], label="<Fz>", linestyle='--', marker='o', color='y')
    ax2.plot(output_gillespie[:,0], output_gillespie[:,4], label="<X>", linestyle='-.', marker='^', color='g')

    #ax1.set_yticks(np.arange(0, ax1.get_ybound()[1], 1))
    #ax2.set_yticks(np.arange(0, ax2.get_ybound()[1], 10))

    ax1.set_xlabel('Motor rest length [nm]')
    ax1.set_ylabel('Mean unbinding force [pN]')
    #ax2.set_xlabel('Angle [degrees]')
    ax2.set_ylabel('Mean Run Length [nm]')

    fig.suptitle(f'{family}: Mean unbinding force <F> and run length <X> as a function of motor rest length')
    ax1.legend(loc='upper left')
    ax1.grid()
    ax2.legend(loc='upper right')
    plt.savefig(f'RL_{radius}r_{family}_{k_t}kt_{n_motors}motors_{n_it}it.png')
    plt.show()

    '''
    output_gillespie = np.loadtxt(data_file_name)
    plt.plot(output_gillespie[:,0], output_gillespie[:,2], label="<Fx>", linestyle='-', marker='+', color='r')
    plt.plot(output_gillespie[:,0], output_gillespie[:,3], label="<Fz>", linestyle='--', marker='o', color='y')
    plt.ylabel('Mean unbinding force [pN]')
    plt.yscale('log')
    plt.xlabel('Radius bead [nm]')
    plt.title(f'{family}: Mean unbinding force <F> and run length <X> as a function of bead radius')
    plt.legend()
    plt.savefig(f'logscale_force_radius_{family}_{k_t}kt_{n_motors}motors_{n_it}it.png')
    plt.show()

    plt.plot(output_gillespie[:,0], output_gillespie[:,4], label="<X>", linestyle='-.', marker='^', color='g')
    plt.ylabel('Mean unbinding force [pN]')
    plt.yscale('log')
    plt.xlabel('Mean Run Length [nm]')
    plt.title(f'{family}: Mean unbinding force <F> and run length <X> as a function of bead radius')
    plt.legend()
    plt.savefig(f'logscale_RL_radius_{family}_{k_t}kt_{n_motors}motors_{n_it}it.png')
    plt.show()
    '''
    return

##### motor figures #####

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

#### bead figures ####

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

####################

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

