import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d

def traj_kt(subdir, family, list_kt, n_motors, time_frame, it=1, file_name=None):
    """

    Parameters
    ----------

    Returns
    -------

    """

    if file_name == None:
        fig_name = f'traj_kt_{family}'
    else:
        fig_name = file_name

    for kt in list_kt:
        # Unpickle motor_0 object
        pickleMotor0 = open(f'..\motor_objects\\kt\{subdir}\motor0_{family}_{kt}kt_{n_motors}motors', 'rb')
        motor0 = pickle.load(pickleMotor0)
        pickleMotor0.close()

        if not os.path.isdir(f'..\motor_objects\\kt\{subdir}\\figures'):
            os.makedirs(f'..\motor_objects\\kt\{subdir}\\figures')
        x = motor0.time_points[it][time_frame[0]:time_frame[1]]
        y = motor0.x_bead[it][time_frame[0]:time_frame[1]]
        plt.step(x, y, where='post')
        plt.scatter(x,y)
        plt.title(f'radius {kt}')
        plt.savefig(f'..\motor_objects\\varvalue\{subdir}\\figures\{fig_name}_{kt}kt_{time_frame}_{it}it.png')
        plt.show()

    return

def traj_team_size(subdir, family, n_motors, time_frame, it=1, file_name=None):
    """

    Parameters
    ----------

    Returns
    -------

    """

    if file_name == None:
        fig_name = f'traj_teamsize_{family}'
    else:
        fig_name = file_name

    for n in range(1, n_motors + 1):
        # Unpickle motor_0 object
        pickleMotor0 = open(f'..\motor_objects\\teamsize\{subdir}\motor0_{family}_{n_motors}motors', 'rb')
        motor0 = pickle.load(pickleMotor0)
        pickleMotor0.close()

        if not os.path.isdir(f'..\motor_objects\\teamsize\{subdir}\\figures'):
            os.makedirs(f'..\motor_objects\\teamsize\{subdir}\\figures')
        x = motor0.time_points[it][time_frame[0]:time_frame[1]]
        y = motor0.x_bead[it][time_frame[0]:time_frame[1]]
        plt.step(x, y, where='post')
        plt.scatter(x,y)
        plt.title(f'Teamsize {n_motors}')
        plt.savefig(f'..\motor_objects\\teamsize\{subdir}\\figures\{fig_name}_{n_motors}motors_{time_frame}_{it}it.png')
        plt.show()

    return

def traj_radii(subdir, family, list_r, n_motors, time_frame, it=1, file_name=None, all_traj=False):
    """

    Parameters
    ----------

    Returns
    -------

    """

    if file_name == None:
        fig_name = f'traj_radii_{family}'
    else:
        fig_name = file_name

    for r in list_r:
        # Unpickle motor_0 object
        pickleMotor0 = open(f'..\motor_objects\\radius\{subdir}\motor0_{family}_{r}radius_{n_motors}motors', 'rb')
        motor0 = pickle.load(pickleMotor0)
        pickleMotor0.close()

        if not os.path.isdir(f'..\motor_objects\\radius\{subdir}\\figures'):
            os.makedirs(f'..\motor_objects\\radius\{subdir}\\figures')

        if all_traj==False:
            x = motor0.time_points[it][time_frame[0]:time_frame[1]]
            y = motor0.x_bead[it][time_frame[0]:time_frame[1]]
            plt.step(x, y, where='post')
            plt.scatter(x,y)
            plt.title(f'radius {r}')
            plt.savefig(f'..\motor_objects\\radius\{subdir}\\figures\{fig_name}_{r}r_{time_frame}_{it}it.png')
            plt.show()
        else:
            x = motor0.time_points
            x_new = np.asanyarray(x, dtype=list)
            y = motor0.x_bead
            y_new = np.asanyarray(y, dtype=list)
            #print(len(x_new[1]))
            #print(len(y_new[1]))
            plt.step(x_new, y_new, where='post')
            plt.scatter(x,y)
            plt.title(f'radius {r}')
            plt.savefig(f'..\motor_objects\\radius\{subdir}\\figures\{fig_name}_{r}r_alltraj.png')
            plt.show()

    return

def traj_restlength(subdir, family, list_rl, n_motors, time_frame, it=1, file_name=None, all_traj=False):
    """

    Parameters
    ----------

    Returns
    -------

    """
    if file_name == None:
        fig_name = f'traj_restlength_{family}'
    else:
        fig_name = file_name

    for rl in list_rl:
        # Unpickle motor_0 object
        pickleMotor0 = open(f'..\motor_objects\\rl\{subdir}\motor0_{family}_{rl}rl{n_motors}motors', 'rb')
        motor0 = pickle.load(pickleMotor0)
        pickleMotor0.close()

        if not os.path.isdir(f'..\motor_objects\\rl\{subdir}\\figures'):
            os.makedirs(f'..\motor_objects\\rl\{subdir}\\figures')

        if all_traj==False:
            x = motor0.time_points[it][time_frame[0]:time_frame[1]]
            y = motor0.x_bead[it][time_frame[0]:time_frame[1]]
            plt.step(x, y, where='post')
            plt.scatter(x,y)
            plt.title(f'rest length {rl}')
            plt.savefig(f'..\motor_objects\\rl\{subdir}\\figures\{fig_name}_{rl}rl_{time_frame}_{it}it.png')
            plt.show()
        else:
            x = motor0.time_points
            y = motor0.x_bead
            plt.step(x, y, where='post')
            plt.title(f'rest length {rl}')
            plt.savefig(f'..\motor_objects\\rl\{subdir}\\figures\{fig_name}_{rl}rl_alltraj.png')
            plt.show()

    return

def traj_sym(subdir, family, it=1, file_name=None, all_traj=False):
    """

    Parameters
    ----------

    Returns
    -------

    """
    if file_name == None:
        fig_name = f'traj_symmetry_{family}'
    else:
        fig_name = file_name

    pickleMotor0 = open(f'..\motor_objects\\symmetry\{subdir}\motor0', 'rb')
    motor0 = pickle.load(pickleMotor0)
    pickleMotor0.close()

    if all_traj == False:
        #motor0_Kinesin-1_0.0kt_1motors.time_points[it].pop()
        x = motor0.time_points[it]
        y = motor0.x_bead[it]
        plt.step(x, y, where='post')
        plt.scatter(x,y)
        plt.title(f'Symmetry bead trajectory')
        plt.savefig(f'..\motor_objects\\symmetry\{subdir}\{fig_name}.png')
        plt.show()
    else:
        x = motor0.time_points
        y = motor0.x_bead
        plt.step(x, y, where='post')
        plt.title(f'Symmetry bead trajectory')
        plt.savefig(f'..\motor_objects\\symmetry\{subdir}\{fig_name}.png')
        plt.show()

    return


### interpolated + noise for experimental-like trajectories ###

def traj_kt_intrpl(subdir, family, n_motors, list_kt, calc_eps, interval=(0, 90), stepsize=0.001, subject='varvalue'):

    """

    Parameters
    ----------

    Returns
    -------

    """

    for kt in list_kt:
        print(f'varvalue={kt}')
        # Unpickle motor0_Kinesin-1_0.0kt_1motors object
        pickle_file_motor0 = open(f'..\motor_objects\\{subject}\{subdir}\motor0_{family}_{kt}kt_{n_motors}motors', 'rb')
        motor0 = pickle.load(pickle_file_motor0)
        pickle_file_motor0.close()

        # Loop through lists in nested list of bead locations
        for index, list_xb in enumerate(motor0.x_bead):
            print(f'index={index}')

            # Original data
            t = motor0.time_points[index]
            t.pop() #not necessary under all conditions, make adjustable + explanation
            #print(len(x))
            x_bead = list_xb
            #print(len(y))
            # Create function
            f = interp1d(t, x_bead, kind='previous')
            # New time points, with chosen step size and interval, within the time range of original data time span
            t_intrpl = np.arange(interval[0], interval[1], stepsize)
            # Do interpolation on new time points
            xbead_intrpl = f(t_intrpl)
            # Create noise
            noise = np.random.normal(loc=0, scale=10, size=(len(t_intrpl),))
            xbead_noise = [x + y for x, y in zip(xbead_intrpl, noise)]
            # Filter original time and location lists to same time interval as t_new
            t_filtered = []
            xbead_filtered = []
            for index, time in enumerate(t):
                if interval[0] <= time <= interval[1]:
                    t_filtered.append(time)
                    xbead_filtered.append(x_bead[index])
            #print(t_filtered)
            #print(xbead_filtered)

            ### Plotting with noise on interpolated lists ###
            plt.plot(t_filtered, xbead_filtered, color='black', label='gillespie Tau')
            plt.scatter(t_filtered, xbead_filtered, color='black')
            plt.plot(t_intrpl, xbead_noise, color='blue', label=f'interpolated {stepsize}s')
            plt.xlabel("Time")
            plt.ylabel("Location")
            plt.title(f"Trajectory with interpolation of {stepsize}s for varvalue={kt} with added noise")
            plt.legend()
            plt.savefig(f'..\motor_objects\{subject}\{subdir}\\figures\\traj_intrpl{stepsize}s_noise_{kt}kt_{calc_eps}.png')
            plt.show()

            ### Without noise ###
            plt.plot(t_filtered, xbead_filtered, color='black', linestyle=':',label='gillespie Tau')
            plt.scatter(t_filtered, xbead_filtered, color='black')
            plt.plot(t_intrpl, xbead_intrpl, color='blue', linestyle='--',label=f'interpolated {stepsize}s')
            plt.scatter(t_intrpl, xbead_intrpl, color='blue')
            plt.xlabel("Time")
            plt.ylabel("Location")
            plt.title(f"Trajectory with interpolation of {stepsize}s for varvalue={kt}")
            plt.legend()
            plt.savefig(f'..\motor_objects\{subject}\{subdir}\\figures\\traj_nr{index}it_intrpl{stepsize}s_{kt}kt_{calc_eps}.png')
            plt.show()

            # Without noise and with step() function on original lists
            plt.step(t_filtered, xbead_filtered, where='post', color='black', linestyle=':',label=f'gillespie Tau')
            plt.scatter(t_filtered, xbead_filtered, color='black')
            plt.plot(t_intrpl, xbead_intrpl, color='blue', linestyle='--', label=f'interpolated {interval}s')
            plt.scatter(t_intrpl, xbead_intrpl, color='blue')
            plt.xlabel("Time")
            plt.ylabel("Location")
            plt.title(f"Trajectory with interpolation of {stepsize}s for varvalue={kt} with plt.step(where=post)")
            plt.legend()
            plt.savefig(f'..\motor_objects\{subject}\{subdir}\\figures\\traj_nr{index}it_intrpl{stepsize}s_{kt}kt_{calc_eps}.png')
            plt.show()

            next_kt = input('next varvalue? yes or no')
            if next_kt == 'yes':
                break
            elif next_kt == 'no':
                continue
            else:
                print('invalid input: yes or no')
                continue

    return

########

def traj_kmratio(dirct, subdir, figname, titlestring, it=0, show=True):
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
    x = motor0.time_points[it][0:750]
    y = motor0.x_bead[it][0:750]
    plt.step(x, y, where='post')
    #plt.scatter(x,y)
    plt.title(f'Trajectory: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_kmr_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


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
    plt.step(x, y, where='post')
    #plt.scatter(x,y)
    plt.title(f'Trajectory: {titlestring}')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_fex_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return
