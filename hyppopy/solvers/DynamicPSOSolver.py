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
        self._add_method("update_param")           # Pass function used to adapt parameters during dynamic PSO as specified by user.
        self._add_method("combine_obj")      # Pass function indicating how to combine objective function arguments and parameters to obtain value.

    def _add_method(self, name, func=None, default=None):
        """
        When designing your child solver class you need to implement the define_interface abstract method where you can
        call _add_member_function to define custom solver options, here of Python callable type, which are automatically 
        converted to class methods.

        :param func: [callable] function object to be passed to solver
        """
        assert isinstance(name, str), "Precondition violation, name needs to be of tyoe str, got {}.".format(type(name))
        if func is not None:
            assert callable(func), "Precondition violation, passed object is not callable!"
        if default is not None:
            assert callable(default), "Precondition violation, passed object is not callable!"
        setattr(self, name, func)
        self._child_members[name] = {"type": "callable", "function": func, "default": default}

    def execute_solver(self, searchspace):
        """
        This function is called immediately after convert_searchspace and uses the output of the latter as input. Its
        purpose is to call the solver libs main optimization function.

        :param searchspace: converted hyperparameter space
        """
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        try:
            tree = optunity.search_spaces.SearchTree(searchspace)   # Set up tree structure to model search space.
            box = tree.to_box()                                     # Create set of box constraints to define given search space.
            f = optunity.functions.logged(self.loss_function)       # Call log here because function signature used later on is internal logic.
            f = tree.wrap_decoder(f)                                # Wrap decoder and constraints for internal search space rep.
            f = optunity.constrainst.wrap_constraints(f, default=sys.float_info.max, range_oo=box)
            """
            'wrap_constraints' decorates function f with given input domain constraints. default [float] gives a 
            function value to default to in case of constraint violations. range_oo [dict] gives open range 
            constraints lb and lu, i.e. lb < x < ub and range = (lb, ub), respectively.
            """
            solver = optunity.make_solver('dynamic particle swarm') # Create solver from given parameters.
            self.best, _ = optunity.optimize(solver=solver,
                                             func=f,
                                             maximize=False,
                                             max_evals=self.max_iterations,
                                             pmap=map,
                                             decoder=tree.decode)
            """
            Use 'optimize' function for respective solver requested:
            optimize(solver,f,maximize=False,max_evals=num_evals,pmap=pmap,decoder=tree.decode)
            In general: optimize(solver,func,maximize=True,max_evals=0,pmap=map,decoder=None)
            Optimize func with given solver.
            :param solver: solver to be used, e.g. result from :func: optunity.make_solver
            :param func: [callable] objective function
            :param maximize: [bool] maximize or minimize
            :param max_evals: [int] maximum number of permitted function evaluations
            :param pmap: [function] map() function to use
            
            :return: solution, named tuple with further details
            optimize function (api.py) internally uses 'optimize' function from requested solver module.
            """
        except Exception as e:
            LOG.error("internal error in optunity.minimize_structured occured. {}".format(e))
            raise BrokenPipeError("internal error in optunity.minimize_structured occured. {}".format(e))
