import motorgillespie.simulation.gs_functions as gsf
import time
import os
import numpy as np

os.environ['PYTHONBREAKPOINT'] = '0'


def gillespie_2D_walk(my_team, motor_0, t_max=100, n_iteration=1000, dimension='1D', single_run=False):
    """

    Parameters
    ----------
    my_team : list of MotorProtein
              Team of motor protein objects
    motor_0 : MotorFixed
              Instance of class 'MotorFixed'.
              Fixed location at x = 0.
              Holds simulation parameters and settings and collects cargo data.
    t_max : int, default = 100
            Duration of one iteration of the Gillespie simulation
    n_iteration : int,  default = 1000
                  Number of Gillespie iterations
    dimension : string,  default = 1D
                  one dimensional(multiple motor option) or two dimensional simulation(only for 1 motor)
    single_run : boolean,  default = True
                  If True, each iteration terminates when the cargo unbinds, creating an amount of cargo runs equal
                  to n_iteration. If False, the run will continue until t_max is reached, allowing more then one cargo run
                  per iteration.

    Returns
    -------
    my_team : list of MotorProtein
              Team of motor protein objects holding lists of forces, unbinding forces and walking distances
    motor_0: MotorFixed
             Fixed motor. Holds attributes that contain data of bead and of the simulation as a whole.
    """

    start = time.time()

    # Checking simulation settings
    print('Checking Gillespie settings...')
    dimension_options = ['1D', '2D']
    if dimension not in dimension_options:
        raise ValueError("Invalid function. Expected one of: %s" % dimension_options)
    # Checking motor parameters
    print('Checking motor parameters...')
    for motor in my_team:
        motor.valid_motor(dimension)

    # Set local variables and necessary class attributes once
    print('Setting local variables...')
    radius = motor_0.radius
    rest_length = motor_0.rest_length
    dp_v1 = motor_0.dp_v1
    dp_v2 = motor_0.dp_v2
    temp = motor_0.temp
    if radius and rest_length is not None:
        motor_0.calc_angle()
    k_t = motor_0.k_t
    x_motor0 = motor_0.x_m
    f_ex = motor_0.f_ex

    # Init once:
    motor_0.init_valid_once(my_team)

    #print(motor_0.time_points)
    ## Do i Gillespie runs ##
    print('Begin simulation...')
    for i in range(0, n_iteration):
        print(f'{i}th iteration')

        # Save data: used during simulation
        list_tau = []
        t = 0

        # Set motors to initial state and update lists
        motor_0.init()
        for motor in my_team:
            motor.init(dimension)

        end_run = False
        # Gillespie run until end time (or optional: when bead unbinds)
        while t < t_max:
            #print(f'{end_run}') #debug
            # Update force
            if dimension == '1D':
                gsf.calc_force_1D(my_team, motor_0, k_t, x_motor0, f_ex, i, t, end_run)
                if end_run is True:
                    end_run = False
            else:
                gsf.calc_force_2D(my_team, motor_0, k_t, rest_length, radius, i)

            # Update stepping- and unbinding rates
            for motor in my_team:
                motor.stepping_rate()
                if dimension == '1D':
                    motor.unbind_rate_1D()
                else:
                    motor.unbind_rate_2D(dp_v1, dp_v2, temp, i)

            # Create lists of rates and corresponding ID's of motors for bookkeeping
            list_rates = []
            list_ids = []
            # Append rates of only the possible events (+ corresponding ID's)
            for motor in my_team:
                id = motor.id
                if motor.unbound:
                    list_rates.append(motor.binding_rate)
                    list_ids.append(id)
                else:
                    list_rates.append(motor.epsilon)
                    list_ids.append(id)
                    list_rates.append(motor.alfa)
                    list_ids.append(id)
            # Sum of rates
            sum_rates = sum(list_rates)
            #motor_0.sum_rates.append(sum_rates)

            # Draw waiting time (tau) and update time
            if sum_rates > 0:
                tau = np.random.exponential(scale=(1/sum_rates))
                list_tau.append(tau)
                motor_0.time_points[i].append(motor_0.time_points[i][-1] + tau)
                t += tau
                if t != motor_0.time_points[i][-1]:
                    print('Something wrong with time keeping')
            else:
                print(f'sum of rates is 0, time is {motor_0.time_points[i][-1]}, ln(time): {len(motor_0.time_points[i])},t={t}')
                break

                #####
            ### EVENT ###
                #####
            # Draw which event will happen
            index = gsf.draw_event(list_rates, sum_rates)
            # For control line
            id_match = 0
            event_match = 0
            # Look up which event corresponds to 'index'
            for motor in my_team:
                if list_ids[index] == motor.id:
                    id_match += 1
                    #motor_0.match_events[i].append(motor.id)
                    if list_rates[index] == motor.epsilon:
                        event_match += 1
                        # Save unbinding data
                        if dimension == '1D':
                            motor.forces_unbind.append(motor.f_current) # 1D
                            motor.run_length.append(motor.xm_rel)
                            if motor.direction == 'anterograde':
                                motor_0.antero_unbinds[i] += 1
                            elif motor.direction == 'retrograde':
                                motor_0.retro_unbinds[i] += 1
                            else:
                                raise AssertionError('Something wrong with motor direction')
                        else:
                            motor.fx_unbind.append(motor.f_x[i][-1]) # 2D
                            motor.fz_unbind.append(motor.f_z[i][-1]) # 2D
                            motor.run_length.append(motor.xm_rel)
                        ### Initiate event ###
                        motor.unbinding_event()
                        #print(f'unbind event at t={t}')
                        #if time_leaps > 1:
                            #print(f'antero bound before this event: {motor_0.antero_motors[i][-1]}')
                            #print(f'retro bound before this event: {motor_0.retro_motors[i][-1]}')
                    elif list_rates[index] == motor.alfa:
                        event_match += 1
                        ### Initiate event ###
                        motor.stepping_event()
                    elif list_rates[index] == motor.binding_rate:
                        event_match += 1
                        ### Initiate event ###
                        if f_ex != 0:
                            if motor_0.antero_motors[i][-1] + motor_0.retro_motors[i][-1] == 0:
                                bead_distance = f_ex/motor.k_m
                                motor_bind = 0 - bead_distance
                                motor.binding_event(motor_bind)
                                #print(f'First binding after cargo unbinding, motor_bind={motor_bind} = bead_distance={bead_distance}')
                            else:
                                motor.binding_event(motor_0.x_bead[i][-1])
                        else:
                            motor.binding_event(motor_0.x_bead[i][-1])
                    else:
                        raise AssertionError('Event index not found')
                else:
                    pass

            # There should be only one match
            if id_match != 1:
                raise AssertionError(f'Motor protein ID should have 1 index match not {id_match}')
            if event_match != 1:
                raise AssertionError(f'Event should have 1 index match not {event_match}')

            # Save all current motor locations
            for motor in my_team:
                motor.x_m_abs[i].append(motor.xm_abs)

            # Save how many motors are bound
            antero_bound = 0
            retro_bound = 0
            for motor in my_team:
                if not motor.unbound:
                    if motor.direction == 'anterograde':
                        antero_bound += 1
                    elif motor.direction == 'retrograde':
                        retro_bound += 1
                    else:
                        raise AssertionError('Something wrong with motor direction')

            motor_0.antero_motors[i].append(antero_bound)
            motor_0.retro_motors[i].append(retro_bound)

            # If there are 0 anterograde and retrograde motors bound, but there was at least one motor bound last iteration, the bead has umbound.
            if antero_bound + retro_bound == 0: #and motor_0.antero_motors[i][-2] + motor_0.retro_motors[i][-2] != 0:
                #print(f'no motors bound happened')
                #print(f'it={i}, t={t}')
                #print(f'bound motors = {motor_0.antero_motors[i][-1] + motor_0.retro_motors[i][-1]}')
                #print(f'bound motors PREVIOUS= {motor_0.antero_motors[i][-2] + motor_0.retro_motors[i][-2]}')
                #for motor in my_team:
                    #print(f'{motor.id}: unbound={motor.unbound}')
                #xm_km_list = [(motor.xm_abs * motor.k_m) for motor in my_team]
                # List of Km's of bound motor proteins in motor team
                #km_list = [motor.k_m for motor in my_team if not motor.unbound]
                #km_list.append(k_t)
                # Calculate position beat/cargo
                #bead_loc = sum(xm_km_list)/sum(km_list)
                #print(f'new beadloc={bead_loc}')
                #print(f'old beadloc aka runlength={motor_0.x_bead[i][-1]}')
                end_run = True
                motor_0.stall_time.append(list_tau[-1]) # Stall time before binding
                motor_0.runlength_bead[i].append(motor_0.x_bead[i][-1]) # Save current bead location; run length
                motor_0.time_unbind[i].append(t)
                #print(f'motor_0.x_bead[i][-1]={motor_0.x_bead[i][-1]}')
                #print(f'motor_0.x_bead[i][0]={motor_0.x_bead[i][0]}')
                #print(motor_0.x_bead[i][-1] - motor_0.x_bead[i][0])
                #print(f'len motor0_runlength{len(motor_0.runlength_bead)}')
                #print(f'last runlength={motor_0.runlength_bead[-1]}')
                # Stop this Gillespie run, if this setting is passed
                #detach = True
                #print(f'detach={detach}')
                if single_run is True:
                    break

    end = time.time()
    print(end-start)

    return my_team, motor_0

