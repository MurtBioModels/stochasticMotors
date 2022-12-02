import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import groupby

def time_scale(dirct):

    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print('PRINT NAME IN SUBDIR')
            print(os.path.join(path,subdir))
            sub_path = os.path.join(path,subdir)
            # loop through motor files
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file == 'parameters.txt':
                        continue
                    if file == 'figures':
                        continue
                    if file == 'data':
                        continue
                    if file == 'motor0':
                        print('motor0')
                        pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\{file}', 'rb')
                        motor0 = pickle.load(pickle_file_motor0)
                        pickle_file_motor0.close()
                        time = motor0.time_points
                        endtime_mean = []
                        len_mean = []
                        for i in time:
                            endtime_mean.append(i[-1])
                            len_mean.append(len(i))
                        print(f'mean endtime = {sum(endtime_mean)/len(endtime_mean)}')
                        print(f'mean len(time) = {sum(len_mean)/len(len_mean)}')

    return



def inspect(dirct):

    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print('PRINT NAME IN SUBDIR')
            print(os.path.join(path,subdir))
            sub_path = os.path.join(path,subdir)
            # loop through motor files
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file == 'parameters.txt':
                        continue
                    if file == 'figures':
                        continue
                    if file == 'data':
                        continue
                    if file == 'motor0':
                        print('motor0')
                        pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\{file}', 'rb')
                        motor0 = pickle.load(pickle_file_motor0)
                        pickle_file_motor0.close()
                        print(f'')
                    else:
                        print('PRINT NAME IN FILES')
                        print(os.path.join(sub_path,file))
                        pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                        motor = pickle.load(pickle_file_motor)
                        pickle_file_motor.close()

                        print(f'motor_id={motor.id}, direction={motor.direction}, km={motor.k_m}')



    return


def inspect(dirct):

    path = f'.\motor_objects\\{dirct}'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir == 'figures':
                continue
            if subdir == 'data':
                continue
            print('PRINT NAME IN SUBDIR')
            print(os.path.join(path,subdir))
            sub_path = os.path.join(path,subdir)
            # loop through motor files
            for root2,subdir2,files2 in os.walk(sub_path):
                for file in files2:
                    if file == 'parameters.txt':
                        continue
                    if file == 'figures':
                        continue
                    if file == 'data':
                        continue
                    if file == 'motor0':
                        print('motor0')
                        pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\{file}', 'rb')
                        motor0 = pickle.load(pickle_file_motor0)
                        pickle_file_motor0.close()
                        print(f'')
                    else:
                        print('PRINT NAME IN FILES')
                        print(os.path.join(sub_path,file))
                        pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                        motor = pickle.load(pickle_file_motor)
                        pickle_file_motor.close()

                        print(f'motor_id={motor.id}, direction={motor.direction}, km={motor.k_m}')



    return


def print_things_radius(family, k_t, n_motors, n_it, list_r):

    for r in list_r:

        # Unpickle motor_0 object
        pickleMotor0 = open(f'Motor0_{family}_{k_t}kt_{r}radius_{n_motors}motors_{n_it}it', 'rb')
        motor0 = pickle.load(pickleMotor0)
        pickleMotor0.close()

        # Unpickle list of motor objects
        pickle_file_team = open(f'MotorTeam_{family}_{k_t}kt_{r}radius_{n_motors}motors_{n_it}it', 'rb')
        motor_team = pickle.load(pickle_file_team)
        pickle_file_team.close()

        print(f'{r}radius: {motor0.angle} angle')
        print(f'{r}radius: {motor0.radius} radius')
        #for motor in motor_team:
         #   print(f'{r}radius, max run length={sorted(motor.run_lengths)}')
          #  print(f'{r}radius, max unbinding fx ={sorted(motor.fx_unbind)}')

    return


def print_things_rl(family, k_t, radius, n_motors, n_it, list_rl):

    for rl in list_rl:

        # Unpickle motor_0 object
        pickleMotor0 = open(f'Motor0_{family}_{k_t}kt_{radius}r_{rl}rl_{n_motors}motors_{n_it}it', 'rb')
        motor0 = pickle.load(pickleMotor0)
        pickleMotor0.close()

        # Unpickle list of motor objects
        pickle_file_team = open(f'MotorTeam_{family}_{k_t}kt_{radius}r_{rl}rl_{n_motors}motors_{n_it}it', 'rb')
        motor_team = pickle.load(pickle_file_team)
        pickle_file_team.close()

        #print(f'{rl}rest length: {motor0_Kinesin-1_0.0kt_1motors.angle} angle')
        #print(f'{rl}rest length: {motor0_Kinesin-1_0.0kt_1motors.rest_length} rest length')
        #nrow = len(motor0_Kinesin-1_0.0kt_1motors.time_points)
        #ncol = len(motor0_Kinesin-1_0.0kt_1motors.time_points[100])
        #print(nrow)
        #print(ncol)
        max_t = []
        for i in range(0,99):
            list = np.asarray(motor0.time_points[i])
            #sort = sorted(list, reverse=True)
            diff = np.diff(list)
            max_t.append(max(diff))
            #print(f'{rl}rest length, max unbinding fx ={sorted(motor.fx_unbind)}')
        print(sorted(max_t))

    return


def print_things_sym(subdir, family, n_motors, kt):


    # Unpickle motor_0 object
    pickleMotor0 = open(f'..\motor_objects\\symmetry\{subdir}\Motor0_{family}_{kt}kt_{n_motors}motors', 'rb')
    motor0 = pickle.load(pickleMotor0)
    pickleMotor0.close()

    # Unpickle list of motor objects
    pickle_file_team = open(f'..\motor_objects\\symmetry\{subdir}\MotorTeam_{family}_{kt}kt_{n_motors}motors', 'rb')
    motor_team = pickle.load(pickle_file_team)
    pickle_file_team.close()

    for motor in motor_team:
        #print(f'motor{motor.id}: {motor.direction} unique epsilon: {np.unique(motor.eps_list)}, unique alfa: {np.unique(motor.alfa_list)}')
        print(f'motor{motor.id}: {motor.direction} np.rand match events: {motor.match_events}')


    plt.scatter(motor_team[0].alfa_list, motor_team[1].alfa_list)
    plt.xlabel(motor_team[0].direction)
    plt.ylabel(motor_team[1].direction)
    plt.grid()
    plt.show()
    diff = [a_i - b_i for a_i, b_i in zip(motor_team[0].alfa_list, motor_team[1].alfa_list)]
    print(f'unqiue differences in stepping rate (per index) {np.unique(diff)}')

    antero_count = 0
    retro_count = 0
    for list in motor0.match_events:
        if list[0] == 'anterograde':
            antero_count += 1
        elif list[0] == 'retrograde':
            retro_count += 1
        else:
            print('what?')
    print(f'antero walks first: {antero_count}, retro walks first: {retro_count}')

    return


def checkinggg(dirct):

        path = f'.\motor_objects\\{dirct}'
        for root, subdirs, files in os.walk(path):
            for subdir in subdirs:
                if subdir == 'figures':
                    continue
                if subdir == 'data':
                    continue
                print('PRINT NAME IN SUBDIR')
                print(os.path.join(path,subdir))
                sub_path = os.path.join(path,subdir)

                # Unpickle test_motor0 object
                pickle_file_motor0 = open(f'.\motor_objects\\{dirct}\\{subdir}\motor0', 'rb')
                motor0 = pickle.load(pickle_file_motor0)
                pickle_file_motor0.close()
                #
                xb = motor0.x_bead
                count_zero = []
                for nested in xb:
                    count_zero.append(nested.count('0'))
                print(f'count zero x_b should be 1000: {count_zero.count(0)}')
                #
                # loop through motor files
                for root2,subdir2,files2 in os.walk(sub_path):
                    for file in files2:
                        if file == 'motor0':
                            continue
                        if file == 'parameters.txt':
                            continue
                        if file == 'figures':
                            continue
                        if file == 'data':
                            continue
                        print('PRINT NAME IN FILES')
                        print(os.path.join(sub_path,file))
                        # Unpickle motor
                        pickle_file_motor = open(f'{sub_path}\\{file}', 'rb')
                        motor = pickle.load(pickle_file_motor)
                        pickle_file_motor.close()
                        #
                        xb_list = motor.x_m_abs
                        test1 = [[x[0], x[1]] for x in xb_list]
                        print(test1)
                        print(f'print nested list first two numbers unique: {np.unique(np.array(test1))}')
                        '''
                        #
                        test2 = []
                        for nested in xb_list:
                            count = [len(list(g[1])) for g in groupby(nested) if g[0]==0]
                            print(count)
                            count_twos = []
                            test2.append(count)
                        '''
