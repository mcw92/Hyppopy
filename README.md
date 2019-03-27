&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ![docs_title_logo](./resources/docs_title_logo.png)
# A Hyper-Parameter Optimization Toolbox
<br>

## What is Hyppopy?

Hyppopy is a python toolbox for blackbox optimization. It's purpose is to offer a unified and easy to use interface to a collection of solver libraries. Solver Hyppopy is providing are:

* [Hyperopt](http://hyperopt.github.io/hyperopt/)
* [Optunity](https://optunity.readthedocs.io/en/latest/user/index.html)
* [Optuna](https://optuna.org/)
* [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
* Randomsearch Solver
* Gridsearch Solver


## Installation

1. clone the [Hyppopy](http:\\github.com) project from Github
2. (create a virtual environment), open a console (with your activated virtual env) and go to the hyppopy root folder
3. ```$ pip install -r requirements.txt```
4. ```$ python setup.py install```


## How to use Hyppopy?

#### The HyppopyProject class

The HyppopyProject class takes care all settings necessary for the solver and your workflow. To setup a HyppopyProject instance we can use a nested dictionary or the classes memberfunctions respectively.

```python
# Import the HyppopyProject class
from hyppopy.HyppopyProject import HyppopyProject

# Create a nested dict with a section hyperparameter. We define a 2 dimensional
# hyperparameter space with a numerical dimension named myNumber of type float and
# a uniform sampling. The second dimension is a categorical parameter of type string.
config = {
"hyperparameter": {
    "myNumber": {
        "domain": "uniform",
        "data": [0, 100],
        "type": "float"
    },
    "myOption": {
        "domain": "categorical",
        "data": ["a", "b", "c"],
        "type": "str"
    }
}}

# Create a HyppopyProject instance and pass the config dict to
# the constructor. Alternatively one can use set_config method.
project = HyppopyProject(config=config)

# To demonstrate the second option we clear the project
project.clear()

# and add the parameter again using the member function add_hyperparameter
project.add_hyperparameter(name="myNumber", domain="uniform", data=[0, 100], dtype="float")
project.add_hyperparameter(name="myOption", domain="categorical", data=["a", "b", "c"], dtype="str")
```

```python
from hyppopy.HyppopyProject import HyppopyProject

# We might have seen a warning: 'UserWarning: config dict had no
# section settings/solver/max_iterations, set default value: 500'
# when executing the example above. This is due to the fact that
# most solvers need a value for a maximum number of iterations.
# To take care of solver settings (there might be more in the future)
# one can set a second section called settings. The settings section
# again is splitted into a subsection 'solver' and a subsection 'custom'.
# When adding max_iterations to the section settings/solver we can change
# the number of iterations the solver is doing. All solver except of the
# GridsearchSolver make use of the value max_iterations.
# The usage of the custom section is demonstrated later.
config = {
"hyperparameter": {
    "myNumber": {
        "domain": "uniform",
        "data": [0, 100],
        "type": "float"
    },
    "myOption": {
        "domain": "categorical",
        "data": ["a", "b", "c"],
        "type": "str"
    }
},
"settings": {
    "solver": {
        "max_iterations": 500
    },
    "custom": {}
}}
project = HyppopyProject(config=config)
```

The settings added are automatically converted to a class member with a prefix_ where prefix is the name of the subsection. One can make use of this feature to build custom workflows by adding params to the custom section. More interesting is this feature when developing your own solver.

```python
from hyppopy.HyppopyProject import HyppopyProject

# Creating a HyppopyProject instance
project = HyppopyProject()
project.add_hyperparameter(name="x", domain="uniform", data=[-10, 10], dtype="float")
project.add_hyperparameter(name="y", domain="uniform", data=[-10, 10], dtype="float")
project.add_settings(section="solver", name="max_iterations", value=300)
project.add_settings(section="custom", name="my_param1", value=True)
project.add_settings(section="custom", name="my_param2", value=42)

print("What is max_iterations value? {}".format(project.solver_max_iterations))
if project.custom_my_param1:
	print("What is the answer? {}".format(project.custom_my_param2))
else:
	print("What is the answer? x")
```

#### The HyppopySolver classes

Each solver is a child of the HyppopySolver class. This is only interesting if you're planning to write a new solver, we will discuss this in the section Solver Development. All solvers we can use to optimize our blackbox function are part of the module 'hyppopy.solver'. Below is a list of all solvers available along with their access key in squared brackets.

* HyperoptSolver [hyperopt]
* OptunitySolver [optunity]
* OptunaSolver [optuna]
* BayesOptSolver [bayesopt]
* RandomsearchSolver [randomsearch]
* GridsearchSolver [gridsearch]

There are two options to get a solver, we can import directly from the hyppopy.solver package or we use the SolverPool class. We look into both options by optimizing a simple function, starting with the direct import case.

```python
# Import the HyppopyProject class
from hyppopy.HyppopyProject import HyppopyProject

# Import the HyperoptSolver class, in this case wh use Hyperopt
from hyppopy.solver.HyperoptSolver import HyperoptSolver

# Our function to optimize
def my_loss_func(x, y):
    return x**2+y**2

# Creating a HyppopyProject instance
project = HyppopyProject()
project.add_hyperparameter(name="x", domain="uniform", data=[-10, 10], dtype="float")
project.add_hyperparameter(name="y", domain="uniform", data=[-10, 10], dtype="float")
project.add_settings(section="solver", name="max_iterations", value=300)

# create a solver instance
solver = HyperoptSolver(project)
# pass the loss function to the solver
solver.blackbox = my_loss_func
# run the solver
solver.run()

df, best = solver.get_results()

print("\n")
print("*"*100)
print("Best Parameter Set:\n{}".format(best))
print("*"*100)
```

The SolverPool is a class keeping track of all solver classes. We have several options to ask the SolverPool for the desired solver. We can add an option called use_solver to our settings/custom section or to the project instance respectively, or we can use the solver access key (see solver listing above) to ask for the solver directly.

```python
# import the SolverPool class
from hyppopy.SolverPool import SolverPool

# Import the HyppopyProject class
from hyppopy.HyppopyProject import HyppopyProject

# Our function to optimize
def my_loss_func(x, y):
    return x**2+y**2

# Creating a HyppopyProject instance
project = HyppopyProject()
project.add_hyperparameter(name="x", domain="uniform", data=[-10, 10], dtype="float")
project.add_hyperparameter(name="y", domain="uniform", data=[-10, 10], dtype="float")
project.add_settings(section="solver", name="max_iterations", value=300)
project.add_settings(section="custom", name="use_solver", value="hyperopt")

# create a solver instance. The SolverPool class is a singleton
# and can be used without instanciating. It looks in the project
# instance for the use_solver option and returns the correct solver.
solver = SolverPool.get(project=project)
# Another option without the usage of the use_solver field would be:
# solver = SolverPool.get(solver_name='hyperopt', project=project)

# pass the loss function to the solver
solver.blackbox = my_loss_func
# run the solver
solver.run()

df, best = solver.get_results()

print("\n")
print("*"*100)
print("Best Parameter Set:\n{}".format(best))
print("*"*100)
```