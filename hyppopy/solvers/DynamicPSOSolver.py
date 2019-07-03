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

import os
import logging
import optunity
from pprint import pformat
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from hyppopy.solvers.HyppopySolver import HyppopySolver
from . import OptunitySolver

class DynamicPSOSolver(OptunitySolver):
    """
    Dynamic PSO HypopPy Solver Class
    """
    """
    The DynamicPSOSolver class definition does not need an .__init__() because it inherits from OptunitySolver and
    .__init__() does not really do anything differently for DynamicPSOSolver than it already does for OptunitySolver.
    This is why one can skip defining it and the .__init__() of the superclass will be called automatically.
    """
    def define_interface(self):
        """
        This function is called when HyppopySolver.__init__ function finished. Child classes need to define their
        individual parameters here by calling _add_member function for each class member variable to be defined.
        Using _add_hyperparameter_signature, the structure of a hyperparameter expected by the solver must be defined.
        Both members and hyperparameter signatures are checked later on, before executing the solver, ensuring
        settings passed fulfil the solver's needs.
        """
        self._add_member("max_iterations", int)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)

    def loss_function_call(self, params):
        """
        This function is called within the function loss_function and encapsulates the actual blackbox function call
        in each iteration. The function loss_function takes care of the iteration driving and reporting, but each solver
        lib might need some special treatment between the parameter set selection and the calling of the actual blackbox
        function, e.g. parameter converting.

        :param params: [dict] hyperparameter space sample e.g. {'p1': 0.123, 'p2': 3.87, ...}

        :return: [float] loss
        """
        for key in params.keys():
            if self.project.get_typeof(key) is int:
                params[key] = int(round(params[key]))
        return self.blackbox(**params)

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
            f = tree.wrap_decoder(f)               # Wrap decoder and constraints for internal search space rep.
            f = optunity.constrainst.wrap_constraints(f, default=sys.float_info.max, range_oo=box)
            """
            'wrap_constraints' decorates function f with given input domain constraints. default [float] gives a 
            function value to default to in case of constraint violations. range_oo [dict] gives open range 
            constraints lb and lu, i.e. lb < x < ub and range = (lb, ub), respectively.
            """
            solver = optunity.make_solver('dynamic particle swarm')
            self.best, _ = optunity.optimize(solver=solver,
                                             func=f,
                                             maximize=False,
                                             max_evals=self.max_iterations,
                                             pmap=map,
                                             decoder=tree.decode)
            """
            Set up tree structure to model search space.
            Create set of box constraints to define given search space.
            Suggest solver to use (default: PSO).
            Create solver from given parameters.
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
            optimize function (api.py) internally uses optiize function from requested solver module.
            """
        except Exception as e:
            LOG.error("internal error in optunity.minimize_structured occured. {}".format(e))
            raise BrokenPipeError("internal error in optunity.minimize_structured occured. {}".format(e))

    def split_categorical(self, pdict):
        """
        This function splits the incoming dict into two parts, categorical only entries and other.

        :param pdict: [dict] input parameter description dict

        :return: [dict],[dict] categorical only, others
        """
        categorical = {}
        uniform = {}
        for name, pset in pdict.items():
            for key, value in pset.items():
                if key == 'domain' and value == 'categorical':
                    categorical[name] = pset
                elif key == 'domain':
                    uniform[name] = pset
        return categorical, uniform

    def convert_searchspace(self, hyperparameter):
        """
        This function gets the unified hyppopy-like parameterspace description as input and, if necessary, should
        convert it into a solver lib specific format. The function is invoked when run is called and what it returns
        is passed as searchspace argument to the function execute_solver.

        :param hyperparameter: [dict] nested parameter description dict e.g. {'name': {'domain':'uniform', 'data':[0,1], 'type':'float'}, ...}

        :return: [object] converted hyperparameter space
        """
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        # split input in categorical and non-categorical data
        cat, uni = self.split_categorical(hyperparameter)
        # build up dictionary keeping all non-categorical data
        uniforms = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == 'data':
                    if len(value2) == 3:
                        uniforms[key] = value2[0:2]
                    elif len(value2) == 2:
                        uniforms[key] = value2
                    else:
                        raise AssertionError("precondition violation, optunity searchspace needs list with left and right range bounds!")

        if len(cat) == 0:
            return uniforms
        # build nested categorical structure
        inner_level = uniforms
        for key, value in cat.items():
            tmp = {}
            optunity_space = {}
            for key2, value2 in value.items():
                if key2 == 'data':
                    for elem in value2:
                        tmp[elem] = inner_level
            optunity_space[key] = tmp
            inner_level = optunity_space
        return optunity_space
