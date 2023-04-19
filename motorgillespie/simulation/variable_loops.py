import motorgillespie.simulation.initiate_motors as im
import motorgillespie.simulation.gillespie_simulation as gsim
import pickle
import os
import time


def init_run(sim_params, gill_set, *motor_params, dirct, subdir, sd=None):
    """


    Parameters
    ----------
    sim_params : dictionary
    gill_set : dictionary
    motor_params : dictionary
    dirct : string
    subdir : string
    sd : string

    Returns
    -------

    None
    """

    n_motors = gill_set['n_motors']
    t_max = gill_set['t_max']
    n_it = gill_set['n_it']
    dimension = gill_set['dimension']
    single_run = gill_set['single_run']

    # Directory created in current working directory
    if not os.path.isdir(f'.\motor_objects\{dirct}'):
        os.makedirs(f'.\motor_objects\{dirct}')
    # Motor team: list of all motors containing own data
    os.makedirs(f'.\motor_objects\{dirct}\{subdir}')

    # Create team of motor proteins
    print('Initiating motor team..')
    motor_team = im.init_mixed_team(n_motors, *motor_params)
    # Create fixed motor
    print('Initiating motor0...')
    motor0 = im.init_motor_0(sim_params)
    # Simulate motor dynamics with Gillespie
    print('Call Gillespie simulation...')
    team_out, motor0_out = gsim.gillespie_2D_walk(my_team=motor_team, motor_0=motor0, t_max=t_max, n_iteration=n_it, dimension=dimension, single_run=single_run)
    print('Done simulating')
    #print(motor0 is motor0_out)
    #print(motor_team is team_out)
    #print(motor0.time_points[0] is motor0_out.time_points[0])
    del motor_team
    del motor0


    print('Pickling motor team...')
    while team_out:
        print(f'{team_out[-1].id}_{team_out[-1].family}_{team_out[-1].direction}')
        pickleTeam = open(f'.\motor_objects\{dirct}\{subdir}\{team_out[-1].id}_{team_out[-1].family}_{team_out[-1].direction}', 'wb')
        pickle.dump(team_out[-1], pickleTeam)
        pickleTeam.close()
        team_out.pop()
    # Motor0/fixed motor: holds data about the cargo
    print('Pickling motor0...')
    pickleMotor0 = open(f'.\motor_objects\{dirct}\{subdir}\motor0', 'wb')
    pickle.dump(motor0_out, pickleMotor0)
    pickleMotor0.close()
    del motor0_out
    #for motor in team_out:
    #    pickleTeam = open(f'.\motor_objects\{dirct}\{subdir}\{motor.id}_{motor.family}_{motor.direction}', 'wb')
    #    pickle.dump(motor, pickleTeam)
    #    pickleTeam.close()


    # Write meta data filename in subdirectory
    print('Saving metadata to .txt filename...')
    with open(f".\motor_objects\{dirct}\{subdir}\parameters.txt", "w") as par_file:
        par_file.write(f"Simulation description: {sd}: \n")
        for index, dict in enumerate(motor_params):
            par_file.write(f"Motor{index+1} parameters: \n")
            for mp, value in dict.items():
                par_file.write(f"{mp}={value}\n")
        par_file.write("Simulation parameters: \n")
        for sp, value in sim_params.items():
            par_file.write(f"{sp}={value}\n")
        par_file.write("Gillespie settings: \n")
        for gs, value in gill_set.items():
            par_file.write(f"{gs}={value}\n")

    print('Done')

    return


def simpar_loop(sim_params, varsimpar, simpar, gill_set, *motor_params, dirct, sd=None):
    """
    This function loops through a tuple of sim_par parameters, and performs a Gillespie with each parameter value.
    This way the effect of certain experimental conditions can be investigated. This function can not be used to vary
    any parameters intrinsic to the motor proteins, nor gillespie settings. To vary motor species,
    individual motor parameters, or vary the number of each species, you can create custom scripts to do this.

    Parameters
    ----------
    sim_params : dictionary
    varsimpar : tuple
    simpar : string
    gill_set : dictionary
    motor_params : dictionary
    dirct : string
            Directory to store all motor objects related to the simulated experiment.
            This directory should have a clear name indicating the goal, preferably the same name as the script creating it.
            This directory is created in the current working directory,
            and contains one or multiple subdirectories containing the actual motor objects.
    subdir : string
             Subdirectory to store the the motor object simulated under specific parameter values.
             Contains pickled motor team, pickled test_motor0_1, meta data filename and figure directory.
             Subdirectory name should contain the value(s) of the varied parameter(s).
             For example, the number of motors is varied, or a specific motor parameter.
    sd : string

    Returns
    -------

    None
    """

    n_motors = gill_set['n_motors']
    n_it = gill_set['n_it']
    t_max = gill_set['t_max']
    dimension = gill_set['dimension']
    sim_par = simpar

    t = time.strftime("%Y%m%d_%H%M%S")

    for sp in varsimpar:
        print(f'{sim_par}= {sp}')

        # Create team of motor proteins
        motor_team = im.init_mixed_team(n_motors, *motor_params)
        # Create fixed motor
        sim_params[sim_par] = sp
        motor0 = im.init_motor_0(sim_params)
        # Simulate motor dynamics with Gillespie
        team_out, motor0_out = gsim.gillespie_2D_walk(my_team=motor_team, motor_0=motor0, t_max=t_max, n_iteration=n_it, dimension=dimension)

        # Directory created in current working directory
        os.makedirs(f'.\motor_objects\{dirct}\{t}_{sp}{sim_par}')
        # Motor team: list of all motors containing own data
        pickleTeam = open(f'.\motor_objects\{dirct}\{t}_{sp}{sim_par}\motorteam', 'wb')
        pickle.dump(team_out, pickleTeam)
        pickleTeam.close()
        # Motor0/fixed motor: holds data about the cargo
        pickleMotor0 = open(f'.\motor_objects\{dirct}\{t}_{sp}{sim_par}\motor0', 'wb')
        pickle.dump(motor0_out, pickleMotor0)
        pickleMotor0.close()

        # Write meta data filename in subdirectory
        with open(f".\motor_objects\{dirct}\{t}_{sp}{sim_par}\parameters.txt", "w") as par_file:
            par_file.write(f"Simulation description: {sd}: \n")
            for index, dict in enumerate(motor_params):
                par_file.write(f"Motor{index+1} parameters: \n")
                for mp, value in dict.items():
                    par_file.write(f"{mp}={value}\n")
            par_file.write("Simulation parameters: \n")
            for sp, value in sim_params.items():
                par_file.write(f"{sp}={value}\n")
            par_file.write("Gillespie settings: \n")
            for gs, value in gill_set.items():
                par_file.write(f"{gs}={value}\n")

        return


