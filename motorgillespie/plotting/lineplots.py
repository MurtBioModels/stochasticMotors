import pickle
import matplotlib.pyplot as plt
import os.path
import numpy as np
import seaborn as sns
import pandas as pd


### Lineplots of average runlength and unbinding force per varval ###

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
