
def calc_fu_analytical(f_s, k_t, x_0):
    if k_t > 0:
        unbinding_force_average = f_s/(1+(f_s/(k_t*x_0)))
    else:
        unbinding_force_average = 0

    return unbinding_force_average


def calc_fu_analytical_2(f_s, k_t, eps_0, v_0, k_m):
    if k_t > 0:
        unbinding_force_average = f_s/(1+((f_s*eps_0)/((k_t*k_m*v_0)/(k_m+k_t))))
    else:
        unbinding_force_average = 0

    return unbinding_force_average


def calc_walk_dist_analytical(f_s, k_t, x_0):

    walk_dist_average = x_0/(1+((k_t*x_0)/f_s))

    return walk_dist_average

#def  calc_walk_dist_analytical_2() >> Ask Florian


def loop_analytical(f_s, list_kt, eps_0, v_0, k_m, x_0, family):
    # Import functions :
    import numpy as np

    # Create array
    analytic_output = np.zeros([len(list_kt), 3])
    counter = 0

    # Calculate <F> for different values of Kt
    for kt in list_kt:
        fu = calc_fu_analytical_2(f_s, kt, eps_0, v_0, k_m)
        walk_dist = calc_walk_dist_analytical(f_s, kt, x_0)
        analytic_output[counter, 0] = kt
        analytic_output[counter, 1] = fu
        analytic_output[counter, 2] = walk_dist
        counter += 1

    # Save array
    np.savetxt(f'AnalyticalPerKt_{family}_{len(list_kt)}nKts', analytic_output, fmt='%1.3f', delimiter='\t')

    return



