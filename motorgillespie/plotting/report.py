import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import motorgillespie.simulation.motor_class as mc


### single motor validation ###
def mean_fu_kt(dirct, figname='', show=True):
    """

    Parameters
    ----------

    Returns
    -------

    """
    #
    kt_list = []
    fu_sim = []
    fu_analytical = []
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
            pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
            motor0 = pickle.load(pickle_file_motor0)
            pickle_file_motor0.close()
            #
            kt = motor0.k_t
            print(f'kt={kt}')

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

                    kt_list.append(kt)
                    fu_mean = sum(motor.forces_unbind)/len(motor.forces_unbind)
                    fu_sim.append(fu_mean)
                    if kt > 0:
                        analytical_fu = (motor.f_s)/(1+((motor.f_s*motor.epsilon_0)/((kt*motor.k_m*(motor.step_size*motor.alfa_0))/(motor.k_m+kt))))
                    else:
                        analytical_fu = 0
                    fu_analytical.append(analytical_fu)
                    #
                    sns.color_palette()
                    sns.set_style("whitegrid")

    print(fu_analytical)
    print(fu_sim)
    if len(kt_list) != len(fu_analytical) and len(kt_list) != len(fu_sim):
        AssertionError(f'something goes wrong fix it')

    #
    if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
        os.makedirs(f'.\motor_objects\\{dirct}\\figures')

    plt.figure()
    plt.plot(np.asarray(kt_list), np.asarray(fu_analytical), color='r', marker='o', label='Analytical')
    plt.plot(np.asarray(kt_list), np.asarray(fu_sim), color='b', marker='x',  label='Simulation')
    plt.xlim(0,max(kt_list)+0.02)
    plt.xticks(np.arange(0,max(kt_list)+0.02,0.02))
    plt.ylim(0,7)
    plt.yticks(np.arange(0,8,1))
    plt.xlabel('Trap stiffness [pN/nm]')
    plt.ylabel('Mean unbinding force [pN]')
    plt.legend(loc='lower right')
    plt.savefig(f'.\motor_objects\{dirct}\\figures\\fu_trap_analytical_{figname}.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'.\motor_objects\\{dirct}\\figures\\fu_trap_analytical_{figname}', format='svg', dpi=300)

    if show == True:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        plt.close()
        print('Figure saved')

    return


def plot_rl_zeroforce(dirct, figname='', show=False):
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

            rl = motor0.runlength_cargo
            flat_rl = [element for sublist in rl for element in sublist]
            #
            sns.color_palette()
            sns.set_style("whitegrid")

            x = np.linspace(min(flat_rl), max(flat_rl), 100)
            y = (0.66/740) * np.exp(-0.66/740*x)
            #
            if not os.path.isdir(f'.\motor_objects\\{dirct}\\figures'):
                os.makedirs(f'.\motor_objects\\{dirct}\\figures')
            plt.figure()
            plt.hist(x=flat_rl, bins=100, density=True, color='gray')
            plt.plot(x, y, color='r')
            plt.xlabel('Run length [nm]')
            plt.ylabel('Probability density')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\distplot_rl_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            plt.savefig(f'.\motor_objects\\{dirct}\\figures\\distplot_rl_{figname}', format='svg', dpi=300)

            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')
    return


def trajectories(dirct, figname='', it=0, show=True):
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
            x = motor0.time_points[it][0:150]
            y = motor0.x_cargo[it][0:150]
            plt.step(x, y, where='post')
            plt.xlabel('Time [s]')
            plt.xticks(np.arange(0, x[-1], step=1))
            plt.ylabel('Displacement [nm]')
            #plt.scatter(x,y)
            plt.title(f'')
            plt.savefig(f'.\motor_objects\{dirct}\\figures\\traj_{motor0.k_t}_{figname}.png', format='png', dpi=300, bbox_inches='tight')
            if show == True:
                plt.show()
                plt.clf()
                plt.close()
            else:
                plt.clf()
                plt.close()
                print('Figure saved')

    return




if __name__ == '__main__':
    f = np.arange(-8, 8.1, 0.01)
    epsilon_0 = 0.66
    f_d = 2.1
    epsilon = epsilon_0 * (np.e**(abs(f)/f_d))


    plt.figure()
    plt.plot(f, step(f_current=f, f_s=7, alfa_0=92.5), color='black')
    plt.title('Force dependent stepping rate', fontsize=18)
    plt.xlabel('Force', fontsize=18)
    plt.ylabel('Stepping rate', fontsize=18)
    plt.xticks([0], fontsize=18)
    plt.yticks([0], fontsize=18)
    plt.ylim(ymin=0)
    plt.xlim(xmin=-8)
    plt.xlim(xmax=8)
    plt.axvline(x=0, color='gray', ls=':')
    plt.text(7, -5, 'Fs', fontsize='x-large')
    plt.text(-9, 92.5, '\u03C30', fontsize='x-large')
    #plt.tight_layout()
    plt.savefig('figure_1.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()


    plt.figure()
    plt.plot(f, epsilon, color='black')
    plt.title('Force dependent unbinding rate', fontsize=18)
    plt.xlabel('Force', fontsize=18)
    plt.ylabel('Unbinding rate', fontsize=18)
    plt.xticks([0], fontsize=18)
    plt.axvline(x=0, color='gray', ls=':')
    plt.vlines(x=[-2.1, 2.1], ymin=[0, 0], ymax=[1.79, 1.79], lw=2, colors='gray', ls=':')
    plt.text(f_d, -2, 'Fd', fontsize='x-large')
    plt.text(-1*f_d, -2, 'Fd', fontsize='x-large')
    plt.yticks([0], fontsize=18)
    plt.ylim(ymin=0)
    plt.xlim(xmin=-8)
    plt.xlim(xmax=8)
    plt.hlines(y=0.66, xmin=-8, xmax=0, color='black', linestyle=':')
    plt.text(-9.5, 0.66, '\u03B50', fontsize='x-large')
    plt.savefig('figure_2.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
