import unittest
from motorgillespie.simulation import motor_class as mc
from motorgillespie.simulation import gillespie_simulation as gs



class TestMotorFixed(unittest.TestCase):

        def setUp(self):

            self.test_motor = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
            f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'bound', id = 0,
            calc_eps = 'exponential')
            self.test_motor0 = mc.MotorFixed(k_t=0, f_ex=0, dp_v1=None, dp_v2=None, radius=None, rest_length=None, temp=None)

        def testForces(self):
            my_team, motor_0 = gs.gillespie_2D_walk(motor_team=[self.test_motor], motor_fixed=self.test_motor0, t_max=100, n_runs=1000, dimension='1D')
