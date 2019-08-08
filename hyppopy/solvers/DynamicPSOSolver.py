# Hyppopy - A Hyper-Parameter Optimization Toolbox
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

"""
*********
* TO DO *
*********
# define_interface(self): How to get names of functions passed by the user?
# Check _add_method function with Markus/Oskar.
"""

import os
import sys
import numpy
import datetime
import logging
import optunity
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL
from hyperopt import Trials

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from hyppopy.solvers.HyppopySolver import HyppopySolver
from .OptunitySolver import OptunitySolver

class DynamicPSOSolver(OptunitySolver):
    """
    Dynamic PSO HyppoPy Solver Class
    """
    """
    The DynamicPSOSolver class definition does not need an .__init__() because it inherits from OptunitySolver and
    .__init__() does not really do anything differently for DynamicPSOSolver than it already does for OptunitySolver.
    This is why one can skip defining it and the .__init__() of the superclass will be called automatically.
    The functions 'loss_function_call', 'split_categorical' and 'convert_space' are not defined
    here as they are inherited from the parent class OptunitySolver without any changes.
    """
    
    def define_interface(self):
        """
        Function called after instantiation to define individual parameters for child solver class by calling
        _add_member function for each class member variable to be defined. When designing your own solver class,
        you need to implement this method to define custom solver options that are automatically converted
        to class attributes.
        """
        super().define_interface()
        self._add_method("update_param")        # Pass function used to adapt parameters during dynamic PSO as specified by user.
        self._add_method("combine_obj")         # Pass function indicating how to combine obj. func. arguments and parameters to obtain scalar value.
        self._add_member("num_args_obj", int)    # Pass number of arguments/terms contributing to obj. func.
        self._add_member("num_params_obj", int)  # Pass number of parameters of obj. func.
        self._add_hyperparameter_signature(name="domain", dtype=str, options=["uniform", "loguniform", "categorical"])

    def _add_method(self, name, func=None, default=None):
        """
        When designing your child solver class you need to implement the define_interface abstract method where you can
        call _add_member_function to define custom solver options, here of Python callable type, which are automatically 
        converted to class methods.

        :param func: [callable] function object to be passed to solver
        """
        assert isinstance(name, str), "Precondition violation, name needs to be of type str, got {}.".format(type(name))
        if func is not None:
            assert callable(func), "Precondition violation, passed object is not callable!"
        if default is not None:
            assert callable(default), "Precondition violation, passed object is not callable!"
        setattr(self, name, func)
        self._child_members[name] = {"type": "callable", "function": func, "default": default}

    def convert_searchspace(self, hyperparameter):
        """
        Get unified hyppopy-like parameter space description as input and, if necessary,
        convert it into a solver-lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':float}, ...}

        :return: [object] converted hyperparameter space
        :return: [dict] dict keeping domains for different hyperparameters.
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        # Split input in categorical and non-categorical data.
        cat, uni = self.split_categorical(hyperparameter)
        # Build up dict keeping all non-categorical data.
        uniforms = {}
        domains = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == "data":
                    if len(value2) == 3:
                        uniforms[key] = value2[0:2]
                    elif len(value2) == 2:
                        uniforms[key] = value2
                    else:
                        raise AssertionError("precondition violation, optunity searchspace needs list with left and right range bounds!")
                if key2 == "domain":
                    domains[key] = value2

        if len(cat) == 0:
            return uniforms, domains
        # Build nested categorical structure.
        inner_level = uniforms
        for key, value in cat.items():
            tmp = {}
            optunity_space = {}
            for key2, value2 in value.items():
                if key2 == "data":
                    for elem in value2:
                        tmp[elem] = inner_level
                if key2 == "domain":
                    domains[key] = value2
            optunity_space[key] = tmp
            inner_level = optunity_space
        return optunity_space, domains

    def execute_solver(self, searchspace, domains):
        """
        This function is called immediately after convert_searchspace and uses the output of the latter as input. Its
        purpose is to call the solver lib's main optimization function.

        :param searchspace: converted hyperparameter space
        """
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        tree = optunity.search_spaces.SearchTree(searchspace)   # Set up tree structure to model search space.
        box = tree.to_box()                                     # Create set of box constraints to define given search space.
        f = optunity.functions.logged(self.loss_function)       # Call log here because function signature used later on is internal logic.
        f = tree.wrap_decoder(f)                                # Wrap decoder and constraints for internal search space rep.
        f = optunity.constraints.wrap_constraints(f, default=sys.float_info.max*numpy.ones(self.num_args_obj), range_oo=box)
        # 'wrap_constraints' decorates function f with given input domain constraints. default [float] gives a 
        # function value to default to in case of constraint violations. range_oo [dict] gives open range 
        # constraints lb and lu, i.e. lb < x < ub and range = (lb, ub), respectively.

        try:
            self.best, _ = optunity.optimize_dyn_PSO(func=f,
                                                     box=box,
                                                     domains=domains,
                                                     maximize=False,
                                                     max_evals=self.max_iterations,
                                                     num_args_obj=self.num_args_obj,
                                                     num_params_obj=self.num_params_obj,
                                                     pmap=map,
                                                     decoder=tree.decode,
                                                     update_param=self.update_param,
                                                     eval_obj=self.combine_obj,   
                                                     )
            """
            optimize_dyn_PSO(func, maximize=False, max_evals=0, pmap=map, decoder=None, update_param=None, eval_obj=None)
            Optimize func with dynamic PSO solver.
            :param func: [callable] objective function
            :param maximize: [bool] maximize or minimize
            :param max_evals: [int] maximum number of permitted function evaluations
            :param pmap: [function] map() function to use
            :param update_param: [function] function to update parameters of objective function
                                 based on current state of knowledge
            :param eval_obj: [function] function giving functional form of objective function, i.e.
                             how to combine parameters and terms to obtain scalar fitness/loss.
            
            :return: solution, named tuple with further details
            optimize_dyn_PSO function (api.py) internally uses 'optimize' function from dynamic PSO solver module.
            """
        except Exception as e:
            LOG.error("Internal error in optunity.optimize_dyn_PSO occured. {}".format(e))
            raise BrokenPipeError("Internal error in optunity.optimize_dyn_PSO occured. {}".format(e))

    def run(self, print_stats=True):
        """
        This function starts the optimization process.
        :param print_stats: [bool] en- or disable console output
        """
        self._idx = 0
        self.trials = Trials()

        start_time = datetime.datetime.now()
        try:
            search_space, domains = self.convert_searchspace(self.project.hyperparameter)
        except Exception as e:
            msg = "Failed to convert searchspace, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
        try:
            self.execute_solver(search_space, domains)
        except Exception as e:
            msg = "Failed to execute solver, error: {}".format(e)
            LOG.error(msg)
            raise AssertionError(msg)
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

