import unittest
from motorgillespie.simulation import motor_class as mc


class TestMotors(unittest.TestCase):


    def test_print(self):


    def test_valid_motor(self):

        test_motor_params = {
         'family': None,
         'member': None,
         'step_size': None,
         'k_m': None,
         'v_0': None,
         'alfa_0': None,
         'f_s': None,
         'epsilon_0': None,
         'f_d': None,
         'bind_rate': None,
         'direction': None,
         'init_state': None,
         'calc_eps': None,
        }
        my_team = mc.MotorProtein(test_motor_params['family'], test_motor_params['member'], test_motor_params['k_m'], test_motor_params['alfa_0'], test_motor_params['f_s'], test_motor_params['epsilon_0'], test_motor_params['f_d'], test_motor_params['bind_rate'], test_motor_params['step_size'], test_motor_params['direction'], test_motor_params['init_state'], test_motor_params['calc_eps'], 0)


    def test_init(self):

    ### Force dependent rate equations ###
    def test_stepping_rate(self):

    def test_unbind_rate_1D(self):

    def test_unbind_rate_2D(self):

    ### Events ###
    def test_stepping_event(self):

    def test_binding_event(self):

    def test_unbinding_event(self):



if __name__ == '__main__':
    unittest.main()
