'''
Created on Jun 14, 2015

@author: Andre Biedenkapp
'''
import unittest
import os

from smac.parameter_importance.epmimportance import EPMImportance

class ImportanceTest(unittest.TestCase):

    def setUp(self):
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(os.path.join(base_directory, '..', '..'))
        self.current_dir = os.getcwd()
        os.chdir(base_directory)

    def tearDown(self):
        os.chdir(self.current_dir)

    # Simple test where a child is active in the beginning but the flip of the parent deactivates the child.
    def test_deactivate_child(self):
        scenario_fn = 'test/test_files/ablation_test/ab_test1_scenario.txt'
        runhistory_fn = 'test/test_files/ablation_test/ab_test_runhistory.json'
        traj_fn = 'test/test_files/ablation_test/ab_test1_traj_aclib2.json'

        epm_imp = EPMImportance(scenario_fn=scenario_fn,
                                runhistory_fn=runhistory_fn, traj_fn=traj_fn)
        res = epm_imp.run('ab')
        self.assertEqual(len(res), 3)  # make sure x2 gets deactivated since it doesn't contribute to the result
        # TODO long_test_name and x1 might have the same importance value (e.g. 0) and therefor this test might fail
        # sine x1 might be chosen over long_test_name. Either adjust the test files so one dominates the other
        # or just test for x3.
        self.assertEqual(list(res.keys()), ['x3', 'long_test_name', 'x1'])  # applicable since res is an ordered dict
        self.assertEqual(epm_imp._cost, 'log')  # run_obj == runtime forces log transform!

    # make sure that x3 and x2 get flipped together since x3 will activate x2 immediately
    def test_conditional_flips(self):
        scenario_fn = 'test/test_files/ablation_test/ab_test2_scenario.txt'
        runhistory_fn = 'test/test_files/ablation_test/ab_test_runhistory.json'
        traj_fn = 'test/test_files/ablation_test/ab_test2_traj_aclib2.json'

        epm_imp = EPMImportance(scenario_fn=scenario_fn,
                                runhistory_fn=runhistory_fn, traj_fn=traj_fn)
        res = epm_imp.run('ab')
        self.assertEqual(len(res), 3)  # make sure x2 gets flipped with x3 since it gets activated by it
        self.assertEqual(list(res.keys()), ['x3, x2', 'long_test_name', 'x1'])  # applicable since res is
                                                                                # an ordered dict
        self.assertEqual(epm_imp._cost, 'log')  # run_obj == runtime forces log transform!

    # all parameters active and no fancy conditionality or forbidden rules
    def test_simple_path(self):
        scenario_fn = 'test/test_files/ablation_test/ab_test3_scenario.txt'
        runhistory_fn = 'test/test_files/ablation_test/ab_test_runhistory.json'
        traj_fn = 'test/test_files/ablation_test/ab_test3_traj_aclib2.json'

        epm_imp = EPMImportance(scenario_fn=scenario_fn,
                                runhistory_fn=runhistory_fn, traj_fn=traj_fn)
        res = epm_imp.run('ab')
        self.assertEqual(len(res), 4)  # all parameters somehow contribute to the result
        self.assertEqual(epm_imp._cost, 'cost')

    # only two parameters differ in incumbent and default
    def test_simple_path(self):
        scenario_fn = 'test/test_files/ablation_test/ab_test4_scenario.txt'
        runhistory_fn = 'test/test_files/ablation_test/ab_test_runhistory.json'
        traj_fn = 'test/test_files/ablation_test/ab_test4_traj_aclib2.json'

        epm_imp = EPMImportance(scenario_fn=scenario_fn,
                                runhistory_fn=runhistory_fn, traj_fn=traj_fn)
        res = epm_imp.run('ab')
        self.assertEqual(len(res), 2)  # x2 doesn't appera and long_test_name is not flipped
        self.assertEqual(epm_imp._cost, 'cost')

if __name__ == "__main__":
    unittest.main()
