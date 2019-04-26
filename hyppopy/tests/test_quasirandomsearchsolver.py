# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical Image Computing.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE

import unittest
import matplotlib.pylab as plt

from hyppopy.solvers.QuasiRandomsearchSolver import *
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.HyppopyProject import HyppopyProject


class QuasiRandomsearchTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_gaussian_ranges(self):
        interval = [0, 10]
        N = 10
        ranges = get_gaussian_ranges(interval[0], interval[1], N)
        gt = [[0, 2.592443381276233],
              [2.592443381276233, 3.673134565097225],
              [3.673134565097225, 4.251586871937128],
              [4.251586871937128, 4.6491509407201],
              [4.6491509407201, 5.000000000000001],
              [5.000000000000001, 5.350849059279902],
              [5.350849059279902, 5.748413128062873],
              [5.748413128062873, 6.326865434902777],
              [6.326865434902777, 7.407556618723769],
              [7.407556618723769, 10.000000000000002]]
        for a, b in zip(ranges, gt):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

        interval = [-100, 100]
        N = 10
        ranges = get_gaussian_ranges(interval[0], interval[1], N)
        gt = [[-100, -48.151132374475345],
              [-48.151132374475345, -26.537308698055508],
              [-26.537308698055508, -14.96826256125745],
              [-14.96826256125745, -7.0169811855980315],
              [-7.0169811855980315, -1.2434497875801753e-14],
              [-1.2434497875801753e-14, 7.016981185598007],
              [7.016981185598007, 14.968262561257426],
              [14.968262561257426, 26.537308698055483],
              [26.537308698055483, 48.151132374475324],
              [48.151132374475324, 99.99999999999997]]
        for a, b in zip(ranges, gt):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

    def test_get_loguniform_ranges(self):
        interval = [1, 1000]
        N = 10
        ranges = get_loguniform_ranges(interval[0], interval[1], N)
        gt = [[1.0, 1.9952623149688797],
              [1.9952623149688797, 3.9810717055349727],
              [3.9810717055349727, 7.943282347242818],
              [7.943282347242818, 15.848931924611136],
              [15.848931924611136, 31.62277660168379],
              [31.62277660168379, 63.095734448019364],
              [63.095734448019364, 125.89254117941677],
              [125.89254117941677, 251.18864315095806],
              [251.18864315095806, 501.1872336272723],
              [501.1872336272723, 999.9999999999998]]
        for a, b in zip(ranges, gt):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

        interval = [1, 10000]
        N = 50
        ranges = get_loguniform_ranges(interval[0], interval[1], N)
        gt = [[1.0, 1.202264434617413],
              [1.202264434617413, 1.4454397707459274],
              [1.4454397707459274, 1.7378008287493756],
              [1.7378008287493756, 2.0892961308540396],
              [2.0892961308540396, 2.51188643150958],
              [2.51188643150958, 3.0199517204020165],
              [3.0199517204020165, 3.6307805477010135],
              [3.6307805477010135, 4.36515832240166],
              [4.36515832240166, 5.248074602497727],
              [5.248074602497727, 6.309573444801933],
              [6.309573444801933, 7.5857757502918375],
              [7.5857757502918375, 9.120108393559098],
              [9.120108393559098, 10.964781961431854],
              [10.964781961431854, 13.18256738556407],
              [13.18256738556407, 15.848931924611136],
              [15.848931924611136, 19.054607179632477],
              [19.054607179632477, 22.908676527677738],
              [22.908676527677738, 27.542287033381676],
              [27.542287033381676, 33.11311214825911],
              [33.11311214825911, 39.810717055349734],
              [39.810717055349734, 47.863009232263856],
              [47.863009232263856, 57.543993733715695],
              [57.543993733715695, 69.18309709189366],
              [69.18309709189366, 83.17637711026713],
              [83.17637711026713, 100.00000000000004],
              [100.00000000000004, 120.22644346174135],
              [120.22644346174135, 144.54397707459285],
              [144.54397707459285, 173.78008287493753],
              [173.78008287493753, 208.92961308540396],
              [208.92961308540396, 251.18864315095806],
              [251.18864315095806, 301.9951720402017],
              [301.9951720402017, 363.0780547701015],
              [363.0780547701015, 436.5158322401662],
              [436.5158322401662, 524.8074602497729],
              [524.8074602497729, 630.9573444801938],
              [630.9573444801938, 758.5775750291845],
              [758.5775750291845, 912.0108393559099],
              [912.0108393559099, 1096.4781961431854],
              [1096.4781961431854, 1318.2567385564075],
              [1318.2567385564075, 1584.8931924611143],
              [1584.8931924611143, 1905.4607179632485],
              [1905.4607179632485, 2290.867652767775],
              [2290.867652767775, 2754.228703338169],
              [2754.228703338169, 3311.3112148259115],
              [3311.3112148259115, 3981.071705534977],
              [3981.071705534977, 4786.300923226385],
              [4786.300923226385, 5754.399373371577],
              [5754.399373371577, 6918.309709189369],
              [6918.309709189369, 8317.63771102671],
              [8317.63771102671, 10000.00000000001]]
        for a, b in zip(ranges, gt):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

    def test_QuasiRandomSampleGenerator(self):
        N_samples = 10*10*10
        axis_data = {"p1": {"domain": "loguniform", "data": [1, 10000], "type": float},
                     "p2": {"domain": "normal", "data": [-5, 5], "type": float},
                     "p3": {"domain": "uniform", "data": [0, 10], "type": float},
                     "p4": {"domain": "categorical", "data": [False, True], "type": bool}}
        sampler = QuasiRandomSampleGenerator(N_samples, 0.1)
        for name, axis in axis_data.items():
            sampler.set_axis(name, axis["data"], axis["domain"], axis["type"])

        for i in range(N_samples):
            sample = sampler.next()
            self.assertTrue(len(sample.keys()) == 4)
            for k in range(4):
                self.assertTrue("p{}".format(k+1) in sample.keys())
            self.assertTrue(1 <= sample["p1"] <= 10000)
            self.assertTrue(-5 <= sample["p2"] <= 5)
            self.assertTrue(0 <= sample["p3"] <= 10)
            self.assertTrue(isinstance(sample["p4"], bool))
        self.assertTrue(sampler.next() is None)

    def test_solver_uniform(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "uniform",
                    "data": [0, 800],
                    "type": float
                },
                "axis_01": {
                    "domain": "uniform",
                    "data": [-1, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "uniform",
                    "data": [0, 10],
                    "type": float
                }
            },
            "max_iterations": 300
        }

        project = HyppopyProject(config)
        solver = QuasiRandomsearchSolver(project)
        vfunc = VirtualFunction()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(0 <= best['axis_00'] <= 800)
        self.assertTrue(-1 <= best['axis_01'] <= 1)
        self.assertTrue(0 <= best['axis_02'] <= 10)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))

    def test_solver_normal(self):
        config = {
            "hyperparameter": {
                "axis_00": {
                    "domain": "normal",
                    "data": [500, 650],
                    "type": float
                },
                "axis_01": {
                    "domain": "normal",
                    "data": [0, 1],
                    "type": float
                },
                "axis_02": {
                    "domain": "normal",
                    "data": [4, 5],
                    "type": float
                }
            },
            "max_iterations": 500,
            }

        solver = QuasiRandomsearchSolver(config)
        vfunc = VirtualFunction()
        vfunc.load_default()
        solver.blackbox = vfunc
        solver.run(print_stats=False)
        df, best = solver.get_results()
        self.assertTrue(500 <= best['axis_00'] <= 650)
        self.assertTrue(0 <= best['axis_01'] <= 1)
        self.assertTrue(4 <= best['axis_02'] <= 5)

        for status in df['status']:
            self.assertTrue(status)
        for loss in df['losses']:
            self.assertTrue(isinstance(loss, float))


if __name__ == '__main__':
    unittest.main()