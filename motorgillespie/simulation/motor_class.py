import numpy as np


class MotorProtein(object):

    ### Motor protein attributes ###
    def __init__(self, family, member, k_m, alfa_0, f_s, epsilon_0, f_d, bind_rate, step_size, direction, init_state, calc_eps, id):
        """

        Parameters
        ----------
        family : string
        member : string
        step_size : float or integer
                    Step size motor (nm)
        k_m : float or integer
             Motor protein stiffness (pN/nm)
        alfa_0 : float or integer
               Zero force stepping rate (1/s), v_0/size_step
        f_s = 7 : float or integer
                Stall force (pN)
        epsilon_0 : tuple of floats or integers or integer
        f_d : float or integer
              Detachment force (pN)
        bind_rate : float or integer
                   Binding rate (1/s)
        direction : string
                    Retrograde or anterograde
        init_state : string
                     Bound or unbound
        calc_eps : string
        id : integer
             Numerical id for bookkeeping during simulation

        Returns
        -------

        """
        ## Motor protein parameters ##
        self.family = family
        self.member = member
        self.k_m = k_m
        self.alfa_0 = alfa_0
        self.f_s = f_s
        self.epsilon_0 = epsilon_0
        self.f_d = f_d
        self.binding_rate = bind_rate
        self.step_size = step_size
        self.direction = direction
        self.init_state = init_state
        self.calc_eps = calc_eps
        self.id = int(id) + 1
        ## Updating variables ##
        self.unbound = None
        self.xm_abs = None
        self.xm_rel = None
        self.f_current = None # for 2D this is f_x
        self.alfa = None
        self.epsilon = None
        ## Motor data ##
        self.x_m_abs = [] # lis of lists
        self.forces = [] # 1D, list of lists
        self.f_x = [] # 2D, list of lists
        self.f_z = [] # 2D, list of lists
        self.run_length = []
        self.forces_unbind = [] # 1D
        self.fx_unbind = [] # 2D
        self.fz_unbind = [] # 2D

    ### Print info ###
    def info(self):
        print(f'Motor number {self.id} is {self.member} member of the Motor Protein family {self.family}')
        return

    ### Check parameter values are correct ###
    def valid_motor(self, dimension):
        """

        Parameters
        ----------


        Returns
        -------

        """
        # Check if the force dependent unbinding relation
        calc_epsilon_options = ['gaussian', 'exponential', 'constant']
        if self.calc_eps not in calc_epsilon_options:
            raise ValueError("Not a valid motor: invalid force-unbinding relation. Expected one of: %s. Equations can be added in MotorProtein code" % calc_epsilon_options)
        # Check initial state
        initial_options = ['bound', 'unbound']
        if self.init_state not in initial_options:
            raise ValueError("Not a valid motor: invalid initial state. Expected one of: %s." % initial_options)
        # Check motor direction
        direction_options = ['anterograde', 'retrograde']
        if self.direction not in direction_options:
            raise ValueError("Not a valid motor: invalid polarity. Expected one of: %s." % direction_options)
        # Check motor parameters
        if self.f_d == 0:
            raise ValueError("Invalid detachment force (Fd) force. Deliminator can not be zero in equations")
        if self.f_s == 0:
            raise ValueError("Invalid stall force (Fs). Deliminator can not be zero in equations")
        # Check if dimensional restrictions are met
        if dimension == '1D':
            if type(self.epsilon_0) != int and type(self.epsilon_0) != float:
                raise ValueError(f"Not a valid motor: zero force unbinding rate should be an integer or float, not {type(self.epsilon_0)}")
        if dimension == '2D':
            if type(self.epsilon_0) != list:
                raise ValueError(f"Not a valid motor: zero force unbinding rate should be a list, not {type(self.epsilon_0)}")
            if len(self.epsilon_0) != 2:
                raise ValueError(f"Not a valid motor: zero force unbinding rate list should contain two values, not {len(self.epsilon_0)}")
        return

    ### Initiate motor to default state every Gillespie run (t=0) ###
    def init(self, dimension):
        """

        Parameters
        ----------


        Returns
        -------

        """
        if self.init_state == 'unbound':
            self.unbound = True
        elif self.init_state == 'bound':
            self.unbound = False
        else:
            raise ValueError("Motor proteins can either be in an bound or unbound (initial) state")

        self.xm_abs = 0
        self.xm_rel = 0
        self.x_m_abs.append([0])

        if dimension == '1D':
            self.forces.append([])
        elif dimension == '2D':
            self.f_x.append([])
            self.f_z.append([])
        else:
            raise ValueError(f"The simulation settings except only 1D or 2D, not {dimension}")

        return

    ### Manipulate motor ###
    def optogenetics(self):
        """

        Parameters
        ----------


        Returns
        -------

        """
        return

    ### Force dependent rate equations ###
    def stepping_rate(self):
        """

        Parameters
        ----------


        Returns
        -------

        """
        f_current = self.f_current
        direction = self.direction
        f_s = self.f_s
        if direction == 'anterograde':
            if f_current < 0:
                self.alfa = self.alfa_0
            elif f_current > f_s:
                self.alfa = 0
            else:
                self.alfa = self.alfa_0 * (1 - (f_current / f_s))
        elif direction == 'retrograde':
            if f_current > 0:
                self.alfa = self.alfa_0
            elif (-1*f_current) > f_s:
                self.alfa = 0
            else:
                self.alfa = self.alfa_0 * (1 - ((-1 * f_current) / f_s))
        else:
            raise ValueError("Transport must be either retrograde or anterograde")

        return

    def unbind_rate_1D(self):

        """

        Parameters
        ----------


        Returns
        -------

        """

        f_current = (self.f_current**2)**0.5
        calc_eps = self.calc_eps
        #
        if calc_eps == 'gaussian':
            self.epsilon = 1 + (6 * np.exp(-(f_current - 2.5) ** 2))
        elif calc_eps == 'exponential':
            self.epsilon = self.epsilon_0 * (np.e**(f_current / self.f_d))
        elif calc_eps == 'constant':
            self.epsilon = self.epsilon_0
        else:
            raise ValueError("Unbinding equation not recognized. Retry or add equation to unbinding_rate_1")

        return

    def unbind_rate_2D(self, dp_v1, dp_v2, T, i):
        """

        Parameters
        ----------
        dp_v1 : numpy array of floats
            Displacement vector strongly bound state [d_1_x, d_1_z]
        dp_v2 : numpy array of floats
            Displacement weakly strongly bound state [d_2_x, d_2_z]
        T : float or integer
            Temperature in Kelvin [K]

        Returns
        -------

        """
        if T == 0:
            raise ValueError("Invalid temperature (T). Deliminator can not be zero")
        # Check if 2d conditions are given
        if len(self.epsilon_0) != 2:
            raise ValueError("Zero force unbinding rate list should contain two values")
        f_v = np.array([-1*self.f_x[i][-1], self.f_z[i][-1]])
        Boltzmann = 1.38064852e-23
        k1 = self.epsilon_0[0] * np.exp((np.dot(f_v, dp_v1)) / (Boltzmann * T))
        k2 = self.epsilon_0[1] * np.exp((np.dot(f_v, dp_v2)) / (Boltzmann * T))

        self.epsilon = k1 * k2 / (k1 + k2)

        return

    ### Events ###
    def stepping_event(self):
        """

        Parameters
        ----------


        Returns
        -------

        """
        step_size = self.step_size
        direction = self.direction
        if direction == 'anterograde':
            self.xm_abs = self.xm_abs + step_size
            self.xm_rel = self.xm_rel + step_size
        elif direction == 'retrograde':
            self.xm_abs = self.xm_abs - step_size
            self.xm_rel = self.xm_rel - step_size
        else:
            raise ValueError("Transport must be either retrograde or anterograde")
        return

    def binding_event(self, x_bead):
        """

        Parameters
        ----------


        Returns
        -------

        """
        self.xm_abs = x_bead
        self.xm_rel = 0
        self.unbound = False

        return

    def unbinding_event(self):
        """

        Parameters
        ----------


        Returns
        -------

        """
        self.xm_abs = 0
        self.xm_rel = 0
        self.unbound = True

        return


class MotorFixed(object):
    """

    Parameters
    ----------

    Returns
    -------

    """
    ## Class attributes ##
    __id = 0
    __unbound = False
    __x_m = 0

    def __init__(self, dp_v1, dp_v2, radius, rest_length, temp, k_t):
        ## Simulation parameters ##
        self.dp_v1 = dp_v1
        self.dp_v2 = dp_v2
        self.radius = radius
        self.rest_length = rest_length
        self.temp = temp
        self.k_t = k_t
        self.angle = None
        ## Bead/simulation data  ##
        self.time_points = [] # list of lists
        self.x_bead = [] # list of lists
        #self.force_bead = [] # list of lists
        self.retro_motors = [] # list of lists
        self.antero_motors = [] # list of lists
        self.bead_unbind_events = [] # list
        #self.match_events = [] # list of lists; for testing simulation
        self.runlength_bead = [] # divide bij k_t to get force
        self.stall_time = []
        #self.sum_rates = [] # for testing simulation

    ### Initiate motor_0 each gillespie run (t=0) ###
    def init(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        self.time_points.append([0])
        self.x_bead.append([])
        #self.force_bead.append([])
        self.antero_motors.append([])
        self.retro_motors.append([])
        self.bead_unbind_events.append(0)
        #self.match_events.append([])

        return

    ### Set angle in 2D simulation ###
    def calc_angle(self):
        self.angle = np.arcsin(self.radius / (self.radius + self.rest_length))

    ### Getters for private attributes ###
    @property
    def x_m(self):
        return self.__x_m

    @property
    def id(self):
        return self.__id

    @property
    def unbound(self):
        return self.__unbound



























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































