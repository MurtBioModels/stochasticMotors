import unittest
from motorgillespie.simulation import initiate_motors as im
from motorgillespie.simulation import motor_class as mc


class Testcalc_force_1D(unittest.TestCase):
    def setUp(self):
        params = {'family': 'testcase','member': None,'step_size': None,'k_m': 0,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': None,'init_state': 'bound','calc_eps': None,}
        self.test_team = im.init_mixed_team(([2]), params)

    def tearDown(self):
        pass
