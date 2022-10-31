import numpy as np


def calc_force_1D(team, motor_0, k_t, x_motor0, i):
    """
    This function updates the current force acting on each  motor protein
    in a team of multiple motors in Gillespie Stochastic Simulation gillespie_motor_team

    Parameters
    ----------
    team : list of MotorProtein
           Team of motor protein objects for gillespie simulation
    motor_0 : MotorProtein
              Instance of MotorProtein class replacing the optical trap.
              Fixed location at x = 0.
              k_m (motor stiffness) attribute should equal k_t (trap stiffness).
    i : integer
        Current iteration for list indexing

    Returns
    -------
    None
    """

    # List of Km_i*Xm_i of motor proteins in motor team
    xm_km_list = [(motor.xm_abs * motor.k_m) for motor in team]

    # List of Km's of bound motor proteins in motor team
    km_list = [motor.k_m for motor in team if not motor.unbound]
    km_list.append(k_t)

    # Calculate position beat/cargo
    bead_loc = sum(xm_km_list)/sum(km_list)
    motor_0.x_bead[i].append(bead_loc)

    if km_list == 0:
        raise ValueError("Invalid sum of Km's. Deliminator can not be zero")

    # Update forces acting on each individual motor protein
    net_force = 0
    for motor in team:
        if motor.unbound:
            f = 0
        else:
            f = motor.k_m * (motor.xm_abs - bead_loc)
        motor.f_current = f
        motor.forces[i].append(f)
        net_force += f

    # Motor0/fixed motor
    f0 = k_t*(x_motor0 - bead_loc)
    motor_0.force_bead[i].append(f0)
    net_force += f0

    # Net force should be zero
    if abs(net_force) > 10**-11:
        print(f'Net force = {net_force}')
        raise AssertionError('Net force on bead should be zero, look for problems in code')

    return


def calc_force_2D(team, motor_0, k_t, rest_length, radius, i):
    """
    This function updates the current force acting on the motor protein (ONLY ONE MOTOR!)
    in the Gillespie Stochastic simulation gillespie_motor_team

    Parameters
    ----------
    team : list of MotorProtein
           Team of motor protein objects for simulation simulation
    motor_0 : MotorFixed
              Instance of 'MotorFixed' class replacing the optical trap.
              Fixed location at x = 0.
    k_t : float
    rest_length : float or integer
    radius : float or integer
    i : integer
        Current iteration for list indexing

    Returns
    -------
    None
    """

    # List of Km_i*Xm_i of motor proteins in motor team
    if len(team) != 1:
        raise ValueError("The 2D simulation is currently only available for 1 motor simulations")
    xm_km_list = [(motor.xm_abs * motor.k_m) for motor in team]
    xm_km_list.append(motor_0.k_t * motor_0.xm_abs)
    # List of Km's of bound motor proteins in motor team
    km_list = [motor.k_m for motor in team if not motor.init_state]
    km_list.append(k_t)
    # Calculate position bead/cargo
    bead_loc = sum(xm_km_list) / sum(km_list)
    motor_0.x_bead[i].append(bead_loc)
    if km_list == 0:
        raise ValueError("Invalid sum of Km's. Deliminator can not be zero")

    # Update forces acting on each individual motor protein
    net_force = 0
    for motor in team:
        breakpoint()
        if motor.unbound:
            motor.f_current = 0
            motor.f_x[i].append(0)
            motor.f_z[i].append(0)
            net_force += motor.f_current
        else:
            fx = motor.k_m * (motor.xm_abs - bead_loc)
            motor.f_current = fx
            motor.f_x[i].append(fx)
            motor.f_z[i].append(  fx / np.sqrt( ( (1 + (rest_length / radius) )**2 ) - 1)  ) # Dit nog checken met de simpele?
            net_force += motor.f_current

    f_fixed = motor_0.k_t*(motor_0.xm_abs - bead_loc)
    #print(f'force motorfixed calculated={f_fixed}, xbead={bead_loc}')
    net_force += f_fixed

    if abs(net_force) > 10**-11:
        print(net_force)
        raise AssertionError('Net force on bead should be zero, look for problems in code')
    return


def draw_event(list_rates, sum_rates):
    """
    This function draws which event will happen within each Gillespie iteration in gillespie_motor_team module.

    Parameters
    ----------
    list_rates : list of floats
                 List of rates of all possible events for all proteins.
                 Rates are updates each Gillespie iteration based on updated current force acting on each motor.
                 List of rates is created each iteration based on bound or unbound state of each motor.
                 Example: 1 bound + 1 unbound: [alfa motor1, epsilon motor1, attachment rate motor2]
    sum_rates: float

    Returns
    -------
    i - 1 : integer
            Index of event to happen
    """
    # Generate random number
    rand = np.random.rand() # >>> save random seed for reproducibility

    # Find event
    i = 0
    probs_sum = 0.0
    while probs_sum <= rand: # < vs <=
        #breakpoint()
        probs_sum += list_rates[i]/sum_rates
        i += 1

    return i - 1


