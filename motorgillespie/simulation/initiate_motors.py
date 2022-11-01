import motorgillespie.simulation.motor_class as mc


def init_motor_team(motor_params, n_motors):
    """

    Parameters
    ----------
    motor_params: dictionary

    n_motors : integer

    Returns
    -------

    """
    # Creates list of motor objects
    my_team = [mc.MotorProtein(motor_params['family'], motor_params['member'], motor_params['k_m'], motor_params['alfa_0'], motor_params['f_s'], motor_params['epsilon_0'], motor_params['f_d'], motor_params['bind_rate'], motor_params['step_size'], motor_params['direction'], motor_params['init_state'], motor_params['calc_eps'], i) for i in range(n_motors)]

    return my_team


def init_motor_0(sim_params):
    """

    Parameters
    ----------


    Returns
    -------

    """

    # Create fixed motor
    motor_0 = mc.MotorFixed(sim_params['dp_v1'], sim_params['dp_v2'], sim_params['radius'], sim_params['rest_length'], sim_params['temp'], sim_params['k_t'])

    return motor_0


def init_mixed_team(mnr, *motor_params):
    """

    Parameters
    ----------
    mnr : list


    Returns
    -------
    mixed_team: list of MotorProtein objects
    """

    mixed_team = []
    for index, params in enumerate(motor_params):
        for i in range(mnr[index]):
            mixed_team.append(mc.MotorProtein(params['family'], params['member'], params['k_m'], params['alfa_0'], params['f_s'], params['epsilon_0'], params['f_d'], params['bind_rate'], params['step_size'], params['direction'], params['init_state'], params['calc_eps'], len(mixed_team)))
            print(f'{mixed_team[-1].id} - {mixed_team[-1].direction}')

    return mixed_team

'''
    mnr: tuple
        This tuple contains the desired number of each motor species, in the order
        that the associated kinesin_params dictionaries are parsed.
        Example: If you want two Kinesin-1 and three Kinesin-3, and first the Kinesin-1 dictionary is parsed
        and the Kinesin-3 dictionary second: tuple = (2,3)

    kinesin_params: dictionary
                Dictionary(ies) of the desired motor species, one dictionary per species.
                The order of the dictionaries influences the mnr argument.
'''
