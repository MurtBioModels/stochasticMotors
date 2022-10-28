import motorgillespie.simulation.gs_functions as gsf
import time
import os
import numpy as np

os.environ['PYTHONBREAKPOINT'] = '0'


def gillespie_2D_walk(my_team, motor_0, t_max=100, n_iteration=100, dimension='1D', one_trace=True):
    """

    Parameters
    ----------
    my_team : list of MotorProtein
              Team of motor protein objects
    motor_0 : MotorFixed
              Instance of class 'MotorFixed' replacing the optical trap.
              Fixed location at x = 0.
              Holds simulation parameters and settings and collects bead data.
    t_max : int, default = 100
             Duration of one iteration of the Gillespie simulation
    n_iteration : int,  default = 100
                  Number of Gillespie iterations
    dimension : string,  default = 1D
                  one dimensional(multiple motor option) or two dimensional simulation(only for 1 motor)


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
        angle = np.arcsin(radius/(radius+rest_length))
        motor_0.angle = np.rad2deg(angle)
    k_t = motor_0.k_t

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

        # Gillespie run until end time (or optional: when bead unbinds)
        while t < t_max:

            # Update force
            if dimension == '1D':
                gsf.calc_force_1D(my_team, motor_0, i)
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
                if motor.__unbound:
                    list_rates.append(motor.binding_rate)
                    list_ids.append(motor.id)
                else:
                    list_rates.append(motor.__epsilon)
                    list_ids.append(motor.id)
                    list_rates.append(motor.__alfa)
                    list_ids.append(motor.id)
            # Sum of rates
            sum_rates = sum(list_rates)
            motor_0.sum_rates.append(sum_rates)

            # Draw waiting time (tau) and update time
            if sum_rates > 0:
                tau = np.random.exponential(scale=(1/sum_rates))
                list_tau.append(tau)
                motor_0.time_points[i].append(motor_0.time_points[i][-1] + tau)
                t += tau
                if t != motor_0.time_points[i][-1]:
                    print('Something wrong with time keeping')
            else:
                print(f'sum of rates is 0, time is {motor_0.time_points[i][-1]}, length {len(motor_0.time_points[i])},t={t}')
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
                    motor_0.match_events[i].append(motor.id)
                    if list_rates[index] == motor.__epsilon:
                        event_match += 1
                        # Save unbinding data
                        if dimension == '1D':
                            motor.forces_unbind.append(motor.__f_current) # 1D
                            motor.run_length.append(motor.__xm_rel)
                        else:
                            motor.fx_unbind.append(motor.f_x[i][-1]) # 2D
                            motor.fz_unbind.append(motor.f_z[i][-1]) # 2D
                            motor.run_length.append(motor.__xm_rel)
                        # Initiate event
                        motor.unbinding_event()
                    elif list_rates[index] == motor.__alfa:
                        event_match += 1
                        # Initiate event
                        motor.stepping_event()
                    elif list_rates[index] == motor.binding_rate:
                        event_match += 1
                        # Initiate event
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
                motor.x_m_abs[i].append(motor.__xm_abs)

            # Save how many motors are bound
            antero_bound = 0
            retro_bound = 0
            for motor in my_team:
                if motor.__unbound == False:
                    if motor.direction == 'anterograde':
                        antero_bound += 1
                    if motor.direction == 'retrograde':
                        retro_bound += 1
            motor_0.antero_motors[i].append(antero_bound)
            motor_0.retro_motors[i].append(retro_bound)

            # Save bead data per time step
            if len(motor_0.antero_motors[i]) > 1:
                # If there are 0 anterograde and retrograde motors bound, but there was at least one motor bound last iteration, the bead has umbound.
                if motor_0.antero_motors[i][-1] + motor_0.retro_motors[i][-1] == 0 and motor_0.antero_motors[i][-2] + motor_0.retro_motors[i][-2] != 0:
                    motor_0.bead_unbind_events[i].append(1) # Unbinding event is appended as 1
                    motor_0.stall_time.append(list_tau[-1]) # Stall time before binding
                    motor_0.runlength_bead.append(motor_0.x_bead[i][-1]) # Save current bead location; run length
                    # Stop this Gillespie run, if this setting is passed
                    if one_trace == True:
                        break
                else:
                    motor_0.bead_unbind_events[i].append(0) # Append 0 if bead does not unbind
            else:
                pass

    end = time.time()
    print(end-start)

    return my_team, motor_0

