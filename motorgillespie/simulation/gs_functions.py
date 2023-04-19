import numpy as np


def calc_force_1D(team, motor_0, k_t, f_ex, t, i, end_run, x_motor0):
    """
    This function updates the current force acting on each  motor protein
    in a team of multiple motors in Gillespie Stochastic Simulation gillespie_motor_team

    Parameters
    ----------
    team : list of MotorProtein
           Team of motor protein objects for gillespie simulation
    motor_0 : `MotorProtein`
              Cargo object
    k_t : float >= 0
        Trap stiffness
    f_ex: float <= 0
        Constant external force
    i : int
        Current iteration for list indexing
    t : int
        Current time
    end_run : bool
             If last time step the cargo detached, end_run == True
    x_motor0 : int == 0
             This is just an extra control as x_motor0 is a private class attribute of FixedMotor

    Returns
    -------
    None
    """
   # for motor in team: debugggggggg
       # print(f'{motor.id}: unbound is {motor.unbound}, xm list: {motor.x_m_abs[i]}, xm={motor.xm_abs}, km={motor.k_m}')

    # List of Km_i*Xm_i of motor proteins in motor team
    xm_km_sum = sum([(motor.xm_abs * motor.k_m) for motor in team if not motor.unbound])
    if f_ex != 0:
        xm_km_sum += f_ex

    # List of Km's of bound motor proteins in motor team
    km_sum = sum([motor.k_m for motor in team if not motor.unbound])
    if k_t != 0:
        km_sum += k_t

    # Calculate position cargo
    if f_ex == 0:
        net_force = 0
        if t == 0: # Start iteration, dependent on initial state parameter cargo can be bound or unbound
            if motor_0.antero_bound[i][-1] == 0 and motor_0.retro_bound[i][-1] == 0:
                cargo_loc = 0
                #print(f't=0 no motors bound happend, xb={cargo_loc}')
            else:
                cargo_loc = xm_km_sum/km_sum
                #print(f't=0 motors BOUND happend, xb={cargo_loc}')
        elif end_run is True: # Cargo detached last time step
            cargo_loc = 0
            #print(f'end_run=True happend, xb={cargo_loc}')
        else:
            cargo_loc = xm_km_sum/km_sum
    else:
        if t == 0: # Start iteration, dependent on initial state parameter cargo can be bound or unbound
            net_force = f_ex
            # Cargo needs to start at X0=0
            cargo_loc = 0
            # Calculate initial distance of cargo to bound motors
            cargo_distance = xm_km_sum/km_sum
            for motor in team:
                if motor.unbound is False:
                    xm = 0 - cargo_distance
                    motor.xm_abs = xm # change initial 0 to correct position
                    motor.x_m_abs[-1].append(xm)
            #print(f't==0, cargo_distance={cargo_distance}')
        elif end_run is True: # Cargo detached last time step
            net_force = 0
            cargo_loc = 0
            #print(f'end_run=True happend, xb={cargo_loc}')
        else:
            net_force = f_ex
            cargo_loc = xm_km_sum/km_sum
            #print(f'else happened, cargo_loc={cargo_loc}')

    # Append cargo location
    motor_0.x_cargo[i].append(cargo_loc)
    #print(f'it{i}: xb={motor_0.x_cargo[i][-1]}')

    # Update forces acting on each individual motor protein
    for motor in team:
        if motor.unbound:
            f = float('nan')
        else:
            f = motor.k_m * (motor.xm_abs - cargo_loc)
            #print(f'f={f}')
            net_force += f

        motor.f_current = f
        motor.forces[i].append(f)


    # Motor0/fixed motor
    if k_t != 0:
        f0 = k_t*(x_motor0 - cargo_loc)
        net_force += f0

    # Net force should be approximately zero)
    if (net_force**2)**0.5 > 10**-10:
        print(f'Net force = {net_force}')
        raise AssertionError('Net force on cargo should be zero, look for problems in code')

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
    # Calculate position cargo
    cargo_loc = sum(xm_km_list) / sum(km_list)
    motor_0.x_cargo[i].append(cargo_loc)
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
            fx = motor.k_m * (motor.xm_abs - cargo_loc)
            motor.f_current = fx
            motor.f_x[i].append(fx)
            motor.f_z[i].append(  fx / np.sqrt( ( (1 + (rest_length / radius) )**2 ) - 1)  ) # Dit nog checken met de simpele?
            net_force += motor.f_current

    f_fixed = motor_0.k_t*(motor_0.xm_abs - cargo_loc)
    #print(f'force motorfixed calculated={f_fixed}, xcargo={cargo_loc}')
    net_force += f_fixed

    if abs(net_force) > 10**-11:
        print(net_force)
        raise AssertionError('Net force on cargo should be zero, look for problems in code')
    return


def draw_event(list_rates, sum_rates):
    """
    This function draws which event will happen within each Gillespie iteration in gillespie_motor_team module.
    Based on: http://be150.caltech.edu/2019/handouts/12_stochastic_simulation_all_code.html
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


