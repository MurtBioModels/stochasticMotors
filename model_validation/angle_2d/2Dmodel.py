import pickle
import matplotlib.pyplot as plt
import os.path
import numpy as np


def fx_eps(family, k_t, n_motors, n_it, list_r):
    """
    Plotting unbinding rate (__epsilon) of motor in optical trap in 2D simulation.
    The angle is fixed in the simulation.

    Parameters
    ----------

    Returns
    -------

    """
    for r in list_r:

        # Unpickle list of motor objects
        pickle_file_team = open(f'MotorTeam_{family}_{k_t}kt_{r}radius_{n_motors}motors_{n_it}it', 'rb')
        motor_team = pickle.load(pickle_file_team)
        pickle_file_team.close()

        dp_v1 = np.array([2.90, 2.25])
        dp_v2 = np.array([0, 0.18])
        eps0_1 = 0.91
        eps0_2 = 7.62
        angle = np.arcsin(r/(r+35))
        print(angle)
        unbinding_tot = []
        fx_tot_list = []

        for fx in range(-8,9,1):

            fz = abs(fx)*np.tan(angle)
            f_v = np.array([fx, fz])

            k1 = eps0_1 * np.exp((np.dot(f_v,dp_v1)) / 4.1)
            k2 = eps0_2 * np.exp((np.dot(f_v,dp_v2)) / 4.1)

            unbinding_tot.append(k1*k2/(k1+k2))
            fx_tot_list.append(fx)

        print(unbinding_tot)
        print(fx_tot_list)

        for motor in motor_team:
            motor_fx = [i*-1 for i in motor.f_x]
            # Plot simulated mean Run Length as a function of motor team size (N motors)
            plt.plot(fx_tot_list, unbinding_tot, label='fx (when fz is also acting)', linestyle=':', marker='^', color='y')
            plt.plot(motor_fx, motor.eps_list, label='simulation', marker='o', color='r')
            plt.xlabel('Horizontal force Fx [pN]')
            plt.ylabel('Unbinding rate [1/s]')
            #plt.yticks(np.arange(0, max(unbinding_tot), step=1))
            plt.grid()
            plt.title(f'Radius {r}: force detachment rate')
            plt.legend()
            plt.show()

    return


def fx_eps_r(family, list_r, rl, n_motors, subject, subdir):
    """
    ???
    Parameters
    ----------

    Returns
    -------

    """
    for radius in list_r:

        # Unpickle list of motor objects
        pickle_file_team = open(f'..\motor_objects\{subject}\{subdir}\motorteam_{family}_{radius}radius_{n_motors}motors', 'rb')
        motor_team = pickle.load(pickle_file_team)
        pickle_file_team.close()

        dp_v1 = np.array([2.90, 2.25])
        dp_v2 = np.array([0, 0.18])
        eps0_1 = 0.91
        eps0_2 = 7.62
        angle = np.arcsin(radius/(radius+rl))
        print(angle)
        unbind_rates = []
        fx_list = []

        for fx in range(-8,9,1):

            fz = abs(fx)*np.tan(angle)
            f_v = np.array([fx, fz])

            k1 = eps0_1 * np.exp((np.dot(f_v,dp_v1)) / 4.1)
            k2 = eps0_2 * np.exp((np.dot(f_v,dp_v2)) / 4.1)

            unbind_rates.append(k1*k2/(k1+k2))
            fx_list.append(fx)

        print(unbind_rates)
        print(fx_list)

        for motor in motor_team:
            motor_fx = [i*-1 for i in motor.f_x]
            # Plot simulated mean Run Length as a function of motor team size (N motors)
            plt.plot(fx_list, unbind_rates, label='fx (when fz is also acting)', linestyle=':', marker='^', color='y')
            plt.plot(motor_fx, motor.eps_list, label='simulation', marker='o', color='r')
            plt.xlabel('Horizontal force Fx [pN]')
            plt.ylabel('Unbinding rate [1/s]')
            #plt.yticks(np.arange(0, max(unbinding_tot), step=1))
            plt.grid()
            plt.title(f'Rest length {radius}: force-detachment rate')
            plt.legend()
            plt.show()

    return


def trying(family, list_r, n_motors, subdir):
    """
    ???
    Parameters
    ----------

    Returns
    -------

    """
    for radius in list_r:

        # Unpickle list of motor objects
        pickle_file_team = open(f'..\motor_objects\\radius\{subdir}\motorteam_{family}_{radius}radius_{n_motors}motors', 'rb')
        motor_team = pickle.load(pickle_file_team)
        pickle_file_team.close()

        if not os.path.isdir(f'..\motor_objects\\radius\{subdir}\\figures'):
            os.makedirs(f'..\motor_objects\\radius\{subdir}\\figures')
        for motor in motor_team:

            plt.plot(motor.f_x, motor.eps_list, label='unbinding rate', linestyle=':', marker='^', color='y')
            plt.xlabel('Horizontal force Fx [pN]')
            plt.ylabel('Unbinding rate [1/s]')
            plt.grid()
            plt.title(f'Radius {radius}: f_x vs detachment rate')
            plt.legend()
            plt.savefig(f'..\motor_objects\\radius\{subdir}\\figures\\fx_eps_{family}_{n_motors}motors')
            plt.show()

    return
