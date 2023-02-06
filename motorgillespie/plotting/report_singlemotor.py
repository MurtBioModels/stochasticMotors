import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle


def plot_fu_motors(dirct, figname, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """
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
            print(os.path.join(path, subdir))
            sub_path = os.path.join(path, subdir)
            #
            print(f'subdir={subdir}')
            # loop through motor files
            for root2, subdir2, files2 in os.walk(sub_path):
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

                    fu = motor.forces_unbind
                    #
                    sns.color_palette()
                    sns.set_style("whitegrid")

                    #x = np.linspace(min(fu), max(fu), 100)
                    #y = (0.66/740) * np.exp(-0.66/740*x)
                    #
                    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
                        os.makedirs(f'.\motor_objects\\{dirct}\\figures')
                    plt.figure()
                    plt.hist(x=fu, bins=100, density=True, color='black')
                    #plt.plot(x, y, color='r')
                    plt.xlabel('Motor unbinding force [pN]')
                    plt.ylabel('Probability density')
                    plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_fu_{figname}.png', format='png', dpi=300, bbox_inches='tight')

                    if show == True:
                        plt.show()
                        plt.clf()
                        plt.close()
                    else:
                        plt.clf()
                        plt.close()
                        print('Figure saved')
    return


def plot_rl(dirct, figname, show=False):
    """

    Parameters
    ----------

    Returns
    -------

    """
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
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()

            rl = motor0.runlength_bead
            #
            sns.color_palette()
            sns.set_style("whitegrid")

            x = np.linspace(min(rl), max(rl), 100)
            y = (0.66/740) * np.exp(-0.66/740*x)
            #
            if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
                os.makedirs(f'.\motor_objects\\{dirct}\\figures')
            plt.figure()
            plt.hist(x=rl, bins=100, density=True, color='gray')
            #plt.plot(x, y, color='r')
            plt.xlabel('Run length [nm]')
            plt.ylabel('Probability density')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')

            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')
    return


def trajectories(dirct, figname, it=0, show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
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
            #
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()

            #motor0.time_points[it].pop()
            x = motor0.time_points[it]
            y = motor0.x_bead[it]
            print(y[-1])
            print(y[-2])
            y.append(0)
            print(y[-1])
            print(y[-2])
            plt.step(x, y, where='post')
            plt.xlabel('Time [s]')
            plt.xticks(np.arange(0, x[-1], step=0.02))
            plt.ylabel('Displacement [nm]')
            #plt.scatter(x,y)
            plt.title(f'')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return

