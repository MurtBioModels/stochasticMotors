import unittest
from motorgillespie.simulation import motor_class as mc


class TestMotorProtein(unittest.TestCase):

    def setUp(self):
        test_motor = mc.MotorProtein('family': 'Kinesin-1', 'member': 'unknown', 'step_size': 8, 'k_m': 0.2, 'v_0': 740, 'alfa_0': 92.5,
        'f_s': 7, 'epsilon_0': [0.66], 'f_d': 2.1, 'bind_rate': 5, 'direction': 'anterograde', 'init_state': '__unbound',
         'calc_eps': 'exponential')




class TestMotorFixed(unittest.TestCase):

    def setUp(self):






'''
def test_print(self):

def test_valid_motor(self):

def test_init(self):

### Force dependent rate equations ###
def test_stepping_rate(self):

def test_unbind_rate_1D(self):

def test_unbind_rate_2D(self):

### Events ###
def test_stepping_event(self):

def test_binding_event(self):

def test_unbinding_event(self):
'''


if __name__ == '__main__':
    unittest.main()
