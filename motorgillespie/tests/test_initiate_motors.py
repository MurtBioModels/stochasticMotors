import unittest
from motorgillespie.simulation import initiate_motors as im
from motorgillespie.simulation import motor_class as mc

class TestMixedTeam(unittest.TestCase):

    def setUp(self):
        params1 = {'family': 'testfam1','member': None,'step_size': None,'k_m': None,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': 'retrograde','init_state': None,'calc_eps': None,}
        params2 = {'family': 'testfam2','member': None,'step_size': None,'k_m': None,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': 'anterograde','init_state': None,'calc_eps': None,}
        params3 = {'family': 'testfam3','member': None,'step_size': None,'k_m': None,'v_0': None,'alfa_0': None,'f_s': None,'epsilon_0': None,'f_d': None,'bind_rate': None,'direction': 'retrograde','init_state': None,'calc_eps': None,}
        self.test_team = im.init_mixed_team((1,2,3), params1, params2, params3)

    def test_isList(self):
        self.assertIsInstance(self.test_team, list)

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
        self.assertTrue(len(self.test_team) == 6)

    def test_order(self):
        self.assertTrue(self.test_team[0].family =='testfam1')
        self.assertTrue(self.test_team[1].family =='testfam2' and self.test_team[2].family =='testfam2')
        self.assertTrue(self.test_team[3].family =='testfam3' and self.test_team[4].family =='testfam3' and self.test_team[5].family =='testfam3')

    def tearDown(self):
        pass

class TestMotorFixed(unittest.TestCase):

        def setUp(self):
            sim_params = {'dp_v1': None, 'dp_v2': None, 'temp': None, 'radius': None, 'rest_length': None, 'k_t': None}
            self.test_motor0 = im.init_motor_0(sim_params)

        def test_isMotorFixed(self):
            self.assertIsInstance(self.test_motor0, mc.MotorFixed)

        def test_notMotorProtein(self):
            self.assertNotIsInstance(self.test_motor0, mc.MotorProtein)
            self.assertNotIsInstance(self.test_motor0, list)

        def tearDown(self):
            pass


if __name__ == '__main__':
    unittest.main()
