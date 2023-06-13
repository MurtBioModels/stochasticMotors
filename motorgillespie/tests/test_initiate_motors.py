import unittest
from motorgillespie.simulation import initiate_motors as im
from motorgillespie.simulation import motor_class as mc

class TestMixedTeam(unittest.TestCase):

    def setUp(self):
        params1 = {'family': 'testfam1','member': None,'step_size': None,'k_m': None,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': 'retrograde','init_state': None,'calc_eps': None,}
        params2 = {'family': 'testfam2','member': None,'step_size': None,'k_m': None,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': 'anterograde','init_state': None,'calc_eps': None,}
        params3 = {'family': 'testfam3','member': None,'step_size': None,'k_m': None,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': 'retrograde','init_state': None,'calc_eps': None,}
        self.test_team = im.init_mixed_team([1,2,3], params1, params2, params3)
        self.test_indiv = im.init_mixed_team([1], params1)

    def test_isList(self):
        self.assertIsInstance(self.test_team, list)
        self.assertIsInstance(self.test_indiv, list)

    def test_lenlist(self):
        self.assertTrue(len(self.test_team), 6)
        self.assertTrue(len(self.test_indiv), 1)

    def test_isMotorProtein(self):
        for i in self.test_team:
            with self.subTest(i=i.id):
                self.assertIsInstance(i, mc.MotorProtein)

    def test_notMotorFixed(self):
        for i in self.test_team:
            with self.subTest(i=i.id):
                self.assertNotIsInstance(i, mc.MotorFixed)

    def test_rightFamily(self):
        testfam1 = [motor for motor in self.test_team if motor.family == 'testfam1']
        testfam2 = [motor for motor in self.test_team if motor.family == 'testfam2']
        testfam3 = [motor for motor in self.test_team if motor.family == 'testfam3']
        self.assertTrue(len(testfam1) == 1)
        self.assertTrue(len(testfam2) == 2)
        self.assertTrue(len(testfam3) == 3)

        testindiv = [motor for motor in self.test_indiv if motor.family == 'testfam1']
        self.assertTrue(len(testindiv) == 1)

    def test_order(self):
        self.assertTrue(self.test_team[0].family =='testfam1')
        self.assertTrue(self.test_team[1].family =='testfam2' and self.test_team[2].family =='testfam2')
        self.assertTrue(self.test_team[3].family =='testfam3' and self.test_team[4].family =='testfam3' and self.test_team[5].family =='testfam3')

    def tearDown(self):
        del self.test_team


class TestMotorFixed(unittest.TestCase):

        def setUp(self):
            sim_params1 = {'k_t': None, 'f_ex': None, 'dp_v1' : 1, 'dp_v2' : 2, 'radius' : 3, 'rest_length' : 4, 'temp' : 5}
            self.test_motor0_1 = im.init_motor_fixed(sim_params1)
            sim_params2 = {'k_t': 10, 'f_ex': 20}
            self.test_motor0_2 = im.init_motor_fixed(sim_params2)

        def test_isMotorFixed(self):
            self.assertIsInstance(self.test_motor0_1, mc.MotorFixed)
            self.assertIsInstance(self.test_motor0_1, object)
            self.assertIsInstance(self.test_motor0_2, mc.MotorFixed)
            self.assertIsInstance(self.test_motor0_2, object)

        def test_notMotorProtein(self):
            self.assertNotIsInstance(self.test_motor0_1, mc.MotorProtein)
            self.assertNotIsInstance(self.test_motor0_1, list)
            self.assertNotIsInstance(self.test_motor0_2, mc.MotorProtein)
            self.assertNotIsInstance(self.test_motor0_2, list)

        def test_attributes(self):
            self.assertTrue(self.test_motor0_1.k_t == None)
            self.assertTrue(self.test_motor0_1.f_ex == None)
            self.assertTrue(self.test_motor0_1.dp_v1 == 1)
            self.assertTrue(self.test_motor0_1.dp_v2 == 2)
            self.assertTrue(self.test_motor0_1.radius == 3)
            self.assertTrue(self.test_motor0_1.rest_length == 4)
            self.assertTrue(self.test_motor0_1.temp == 5)

            self.assertTrue(self.test_motor0_2.k_t == 10)
            self.assertTrue(self.test_motor0_2.f_ex == 20)
            self.assertTrue(self.test_motor0_2.dp_v1 == None)
            self.assertTrue(self.test_motor0_2.dp_v2 == None)
            self.assertTrue(self.test_motor0_2.radius == None)
            self.assertTrue(self.test_motor0_2.rest_length == None)
            self.assertTrue(self.test_motor0_2.temp ==  None)

        def test_private_attributes(self):
            with self.assertRaises(AttributeError):
                self.test_motor0_1.id = 1
                self.test_motor0_1.x_m = True
                self.test_motor0_1.x_m = 1

        def tearDown(self):
            del self.test_motor0_1
            del self.test_motor0_2


if __name__ == '__main__':
    unittest.main()
