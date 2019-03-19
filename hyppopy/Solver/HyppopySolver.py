# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical and Biological Informatics.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE.txt or http://www.mitk.org for details.
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import abc

import os
import types
import logging
import datetime
import numpy as np
import pandas as pd
from ..globals import DEBUGLEVEL
from ..HyppopyProject import HyppopyProject
from ..BlackboxFunction import BlackboxFunction
from ..VirtualFunction import VirtualFunction

from hyppopy.globals import DEBUGLEVEL, DEFAULTITERATIONS

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


class HyppopySolver(object):

    def __init__(self, project=None):
        self._best = None
        self._trials = None
        self._blackbox = None
        self._max_iterations = None
        self._project = project
        self._total_duration = None
        self._solver_overhead = None
        self._time_per_iteration = None
        self._accumulated_blackbox_time = None

    @abc.abstractmethod
    def execute_solver(self, searchspace):
        raise NotImplementedError('users must define execute_solver to use this class')

    @abc.abstractmethod
    def convert_searchspace(self, hyperparameter):
        raise NotImplementedError('users must define convert_searchspace to use this class')

    def run(self, print_stats=True):
        if 'solver_max_iterations' not in self.project.__dict__:
            msg = "Missing max_iteration entry in project, use default {}!".format(DEFAULTITERATIONS)
            LOG.warning(msg)
            print("WARNING: {}".format(msg))
            setattr(self.project, 'solver_max_iterations', DEFAULTITERATIONS)
        self._max_iterations = self.project.solver_max_iterations

        start_time = datetime.datetime.now()
        try:
            self.execute_solver(self.convert_searchspace(self.project.hyperparameter))
        except Exception as e:
            raise e
        end_time = datetime.datetime.now()
        dt = end_time - start_time
        days = divmod(dt.total_seconds(), 86400)
        hours = divmod(days[1], 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)
        milliseconds = divmod(seconds[1], 0.001)
        self._total_duration = [int(days[0]), int(hours[0]), int(minutes[0]), int(seconds[0]), int(milliseconds[0])]
        if print_stats:
            self.print_best()
            self.print_timestats()

    def get_results(self):
        results = {'duration': [], 'losses': []}
        pset = self.trials.trials[0]['misc']['vals']
        for p in pset.keys():
            results[p] = []

        for n, trial in enumerate(self.trials.trials):
            t1 = trial['book_time']
            t2 = trial['refresh_time']
            results['duration'].append((t2 - t1).microseconds / 1000.0)
            results['losses'].append(trial['result']['loss'])
            losses = np.array(results['losses'])
            results['losses'] = list(losses)
            pset = trial['misc']['vals']
            for p in pset.items():
                results[p[0]].append(p[1][0])
        return pd.DataFrame.from_dict(results), self.best

    def print_best(self):
        print("\n")
        print("#" * 40)
        print("###       Best Parameter Choice      ###")
        print("#" * 40)
        for name, value in self.best.items():
            print(" - {}\t:\t{}".format(name, value))
        print("\n - number of iterations\t:\t{}".format(self.trials.trials[-1]['tid']+1))
        print(" - total time\t:\t{}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
                                                              self._total_duration[1],
                                                              self._total_duration[2],
                                                              self._total_duration[3],
                                                              self._total_duration[4]))
        print("#" * 40)

    def compute_time_statistics(self):
        dts = []
        for trial in self._trials.trials:
            if 'book_time' in trial.keys() and 'refresh_time' in trial.keys():
                dt = trial['refresh_time'] - trial['book_time']
                dts.append(dt.total_seconds())
        self._time_per_iteration = np.mean(dts) * 1e3
        self._accumulated_blackbox_time = np.sum(dts) * 1e3
        tmp = self.total_duration - self._accumulated_blackbox_time
        self._solver_overhead = int(np.round(100.0 / self.total_duration * tmp))

    def print_timestats(self):
        print("\n")
        print("#" * 40)
        print("###        Timing Statistics        ###")
        print("#" * 40)
        print(" - per iteration: {}ms".format(int(self.time_per_iteration*1e4)/10000))
        print(" - total time: {}d:{}h:{}m:{}s:{}ms".format(self._total_duration[0],
                                                           self._total_duration[1],
                                                           self._total_duration[2],
                                                           self._total_duration[3],
                                                           self._total_duration[4]))
        print(" - solver overhead: {}%".format(self.solver_overhead))
        print("#" * 40)

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, value):
        if not isinstance(value, HyppopyProject):
            msg = "Input error, project_manager of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._project = value

    @property
    def blackbox(self):
        return self._blackbox

    @blackbox.setter
    def blackbox(self, value):
        if isinstance(value, types.FunctionType) or isinstance(value, BlackboxFunction) or isinstance(value, VirtualFunction):
            self._blackbox = value
        else:
            self._blackbox = None
            msg = "Input error, blackbox of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        if not isinstance(value, dict):
            msg = "Input error, best of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        self._best = value

    @property
    def trials(self):
        return self._trials

    @trials.setter
    def trials(self, value):
        self._trials = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int):
            msg = "Input error, max_iterations of type: {} not allowed!".format(type(value))
            LOG.error(msg)
            raise IOError(msg)
        if value < 1:
            msg = "Precondition violation, max_iterations < 1!"
            LOG.error(msg)
            raise IOError(msg)
        self._max_iterations = value

    @property
    def total_duration(self):
        return (self._total_duration[0] * 86400 + self._total_duration[1] * 3600 + self._total_duration[2] * 60 + self._total_duration[3]) * 1000 + self._total_duration[4]

    @property
    def solver_overhead(self):
        if self._solver_overhead is None:
            self.compute_time_statistics()
        return self._solver_overhead

    @property
    def time_per_iteration(self):
        if self._time_per_iteration is None:
            self.compute_time_statistics()
        return self._time_per_iteration

    @property
    def accumulated_blackbox_time(self):
        if self._accumulated_blackbox_time is None:
            self.compute_time_statistics()
        return self._accumulated_blackbox_time
