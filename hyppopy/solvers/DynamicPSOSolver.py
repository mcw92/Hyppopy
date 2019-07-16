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
import logging
import optunity
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL

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
        self._add_method("update_param")     # Pass function used to adapt parameters during dynamic PSO as specified by user.
        self._add_method("combine_obj")      # Pass function indicating how to combine objective function arguments and parameters to obtain value.

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

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and uses the output of the latter as input. Its
        purpose is to call the solver lib's main optimization function.

        :param searchspace: converted hyperparameter space
        """
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        try:
            tree = optunity.search_spaces.SearchTree(searchspace)   # Set up tree structure to model search space.
            box = tree.to_box()                                     # Create set of box constraints to define given search space.
            f = optunity.functions.logged(self.loss_function)       # Call log here because function signature used later on is internal logic.
            f = tree.wrap_decoder(f)                                # Wrap decoder and constraints for internal search space rep.
            f = optunity.constraints.wrap_constraints(f, default=sys.float_info.max, range_oo=box)
            """
            'wrap_constraints' decorates function f with given input domain constraints. default [float] gives a 
            function value to default to in case of constraint violations. range_oo [dict] gives open range 
            constraints lb and lu, i.e. lb < x < ub and range = (lb, ub), respectively.
            """
            solver = optunity.make_solver("particle swarm")
            print("Normal PSO solver yay.")
            solver_dyn = optunity.make_solver("dynamic particle swarm")
            self.best, _ = optunity.optimize_dyn_PSO(func=f,
                                                     maximize=False,
                                                     max_evals=self.max_iterations,
                                                     pmap=map,
                                                     decoder=tree.decode,
                                                     update_param=self.update_param,
                                                     eval_obj=self.combine_obj   
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
