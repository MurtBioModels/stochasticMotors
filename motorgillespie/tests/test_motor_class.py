import unittest
from motorgillespie.simulation import motor_class as mc
import math

class TestMotorProtein(unittest.TestCase):

    def setUp(self):
        # Wrong direction
        self.test_motor10 = mc.MotorProtein(family ='Kinesin-1', member ='unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
                                            f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'TEST', init_state = 'unbound', id = 0,
                                            calc_eps = 'exponential')
        # Wrong init_state
        self.test_motor2 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'TEST', id = 0,
        calc_eps = 'exponential')
        # Wrong calc_eps
        self.test_motor8 = mc.MotorProtein(family ='Kinesin-1', member ='unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
                                           f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'unbound', id = 0,
                                           calc_eps = 'TEST')
        # unbound + start_pos
        self.test_motor4 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'unbound', id = 0,
        calc_eps = 'exponential', init_pos=8)
        # Wrong f_s
        self.test_motor9 = mc.MotorProtein(family ='Kinesin-1', member ='unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
                                           f_s = 0, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'unbound', id = 0,
                                           calc_eps = 'exponential')
        # Wrong f_d
        self.test_motor6 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 0, bind_rate = 5, direction = 'anterograde', init_state = 'unbound', id = 0,
        calc_eps = 'exponential')
        # Wrong epsilon_0
        self.test_motor7 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = [0.66, 0.66], f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'unbound', id = 0,
        calc_eps = 'exponential')
        # Valid, unbound
        self.test_motor8 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'bound', id = 0,
        calc_eps = 'exponential')
        # Valid, bound, anterograde
        self.test_motor9 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'bound', id = 0,
        calc_eps = 'exponential')
        # Valid, bound, anterograde + start_pos
        self.test_motor10 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'anterograde', init_state = 'bound', id = 0,
        calc_eps = 'exponential')
        # Valid, bound + retrograde
        self.test_motor11 = mc.MotorProtein(family = 'Kinesin-1', member = 'unknown', step_size = 8, k_m = 0.2, alfa_0 = 92.5,
        f_s = 7, epsilon_0 = 0.66, f_d = 2.1, bind_rate = 5, direction = 'retrograde', init_state = 'bound', id = 0,
        calc_eps = 'exponential')

    def test_valid_motor(self):
        with self.assertRaises(ValueError) as er1:
            self.test_motor10.valid_motor('1D')
        with self.assertRaises(ValueError) as er2:
            self.test_motor2.valid_motor('1D')
        with self.assertRaises(ValueError) as er3:
            self.test_motor8.valid_motor('1D')
        with self.assertRaises(ValueError) as er4:
            self.test_motor4.valid_motor('1D')
        with self.assertRaises(ValueError) as er5:
            self.test_motor9.valid_motor('1D')
        with self.assertRaises(ValueError) as er6:
            self.test_motor6.valid_motor('1D')
        with self.assertRaises(ValueError) as er7:
            self.test_motor7.valid_motor('1D')

        direction_options = ['anterograde', 'retrograde']
        self.assertEqual(str(er1.exception), "Not a valid motor: invalid polarity. Expected one of: %s." % direction_options)
        initial_options = ['bound', 'unbound']
        self.assertEqual(str(er2.exception), "Not a valid motor: invalid initial state. Expected one of: %s." % initial_options)
        calc_epsilon_options = ['gaussian', 'exponential', 'constant']
        self.assertEqual(str(er3.exception), "Not a valid motor: invalid force-unbinding relation. Expected one of: %s. Equations can be added in MotorProtein code" % calc_epsilon_options)
        self.assertEqual(str(er4.exception), "An initially unbound motor cannot have a initial position" )
        self.assertEqual(str(er5.exception), "Invalid detachment force (Fd) force. Deliminator can not be zero in equations" )
        self.assertEqual(str(er6.exception), "Invalid stall force (Fs). Deliminator can not be zero in equations" )
        self.assertEqual(str(er7.exception), f"Not a valid motor: zero force unbinding rate should be an integer or float, not {type(self.test_motor7.epsilon_0)}")

    def test_init(self):

        self.test_motor8.init('1D')
        self.assertTrue(self.test_motor8.unbound, True)
        self.assertTrue(math.isnan(self.test_motor8.xm_abs))
        self.assertTrue(math.isnan(self.test_motor8.xm_rel))
        self.assertTrue(math.isnan(self.test_motor8.x_m_abs[-1][-1]))

        self.test_motor9.init('1D')
        self.assertEqual(self.test_motor9.unbound, False)
        self.assertEqual(self.test_motor9.xm_abs, 0)
        self.assertEqual(self.test_motor9.xm_rel, 0)
        self.assertEqual(self.test_motor9.x_m_abs[-1][-1], 0)

        self.test_motor10.init('1D')
        self.assertEqual(self.test_motor10.unbound, False)
        self.assertEqual(self.test_motor10.xm_abs, 8)
        self.assertEqual(self.test_motor10.xm_rel, 0)
        self.assertEqual(self.test_motor10.x_m_abs[-1][-1], 8)



    ### Force dependent rate equations ###
    def test_stepping_rate(self):
        self.test_motor9.f_current = 8
        self.test_motor9.stepping_rate()
        self.assertEqual(self.test_motor9.alfa, 0)
        self.test_motor9.f_current = -1
        self.test_motor9.stepping_rate()
        self.assertEqual(self.test_motor9.alfa, self.test_motor9.alfa_0)
        self.test_motor9.f_current = 3.5
        self.test_motor9.stepping_rate()
        self.assertEqual(self.test_motor9.alfa, self.test_motor9.alfa_0*0.5)

        self.test_motor11.f_current = -8
        self.test_motor11.stepping_rate()
        self.assertEqual(self.test_motor9.alfa, 0)
        self.test_motor11.f_current = 1
        self.test_motor11.stepping_rate()
        self.assertEqual(self.test_motor9.alfa, self.test_motor9.alfa_0)
        self.test_motor11.f_current = -3.5
        self.test_motor11.stepping_rate()
        self.assertEqual(self.test_motor9.alfa, self.test_motor9.alfa_0*0.5)


    def test_unbind_rate_1D(self):

    def test_unbind_rate_2D(self):

    ### Events ###
    def test_stepping_event(self):

    def test_binding_event(self):

    def test_unbinding_event(self):



#class TestMotorFixed(unittest.TestCase):

   # def setUp(self):





if __name__ == '__main__':
    unittest.main()
