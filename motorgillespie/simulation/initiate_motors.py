import motorgillespie.simulation.motor_class as mc


def init_motor_0(sim_params):
    """

    Parameters
    ----------


    Returns
    -------

    """
    # Create fixed motor
    if all(k in sim_params for k in ("dp_v1", "dp_v2", 'radius', 'rest_length', 'temp')):
        motor_0 = mc.MotorFixed(sim_params['k_t'],  sim_params['f_ex'], sim_params['dp_v1'], sim_params['dp_v2'], sim_params['radius'], sim_params['rest_length'], sim_params['temp'])
    else:
        motor_0 = mc.MotorFixed(sim_params['k_t'],  sim_params['f_ex'])
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
        that the associated plus_params dictionaries are parsed.
        Example: If you want two Kinesin-1 and three Kinesin-3, and first the Kinesin-1 dictionary is parsed
        and the Kinesin-3 dictionary second: tuple = (2,3)

    plus_params: dictionary
                Dictionary(ies) of the desired motor species, one dictionary per species.
                The order of the dictionaries influences the mnr argument.
'''
