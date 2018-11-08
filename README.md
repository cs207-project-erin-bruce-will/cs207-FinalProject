[![Build Status](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject)

[![Coverage Status](https://coveralls.io/repos/github/cs207-project-erin-bruce-will/cs207-FinalProject/badge.svg)](https://coveralls.io/github/cs207-project-erin-bruce-will/cs207-FinalProject)

Erin, Bruce, and Will

# Milestone 2 Update



## Introduction
Autodiff finds the derivatives of a function (to machine precision!) at the same time it finds the value of the function.
```
import autodiff as ad

x = ad.DualNumber('x', 2)
y = ad.DualNumber('y', 3)

out = x/y
out.value # 0.66666
out.derivatives #{x: 1/3, y: -2/(3**2)}
```
Autodiff works across any number of inputs

## How to use autodiff
Autodiff is installed by downloading from [github](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject). Becuase it has no dependencies, you can simply add its folder to your python path (```import sys
sys.path.insert(0, '/path_to_autodiff/')```) and import as normal. 

Autodiff will be available via pip soon.

Using autodiff is very simple:
```python
import autodiff as ad

def f(a,b):
	return a/b*ad.sin(a*b)

out = f(DualNumber('x',2),DualNumber('y',3)))

out.value
-0.416146837

out.derivatives['x']
-1.09115691

out.derivatives['y']
-8.63832555
```

Autodif works by defining dual numbers and then computing as usual. Built-in python operations are handled seamlessley, and operations imported from the math package (e.g. math.sin) must be replaced with thier Autodif equivalents (e.g ad.sin in the example above). Any function that is passed dual numbers as input will return an Autodif DualNumber object that stores the function's value and derivatives at the given point. Scalars are automatically promoted to DualNumbers as needed.

In milestone 2 we intend to deliver a parser that will assist novice users, either by upgrading exsiting code to work with autograd, or helping users write autograd-ready functions via a GUI.


## Background
Automatic differentiation (AD) relies on the chain rule to perform elementary derivatives at the same time it performs the elementary operations which make up the function. Elementary arithemetic operations (such as add, subtract, multiply, divide, etc.) and elementary functions (such as exp, sine, cosine, etc.) are performed at each node, and AD provides the instruction to take the derivate of the given elementary operation or function. These derivatives are accumulated via the chain rule to return the full derivative of a function. There are two modes employed: forward mode, which performs differentiation based on the independent or predictor variable, and reverse mode, which performs differentiation based on the dependent or response variable.

Automatic differentiation evaluates our function at each node, then stores that value to be evaluated at the next node. It recursively applies basic mathematical functions to find the value and the derivatives for more complex equations. An example of this stepwise implementation is shown in the chart below.

| Trace | Elementary Function | Current Value | Elementary Function<BR> Derivative | Elementary Function Derivative Value | 
| :---: | :-----------------: | :-----------: | :----------------------------: | :--------------------------------------------------------: | 
| $x_{1}$ | $x_{1}$ | 3 | $\dot{x}_{1}$ | $1$ |
| $x_{2}$ | $x_{1}^2$ | $9$ | $2x_{1}\dot{x}_{1}$ | $6$ |
| $x_{3}$ | $\sin(x_{2})$ | $\sin(9)$ | $\cos(x_{2})\dot{x}_{2}$ | $6\cos(9)$ |
| $x_{4}$ | $\frac{x_{1}}{2}$ | $\frac{3}{2}$ | $\frac{\dot{x}_{1}}{2}$ | $\frac{1}{2}$ |
| $x_{5}$ | $\cos(x_{1})$ | $\cos(3)$ | $-\sin(x_{1})\dot{x}_{1}$ | $-\sin(3)$ |
| $x_{6}$ | $x_{5}^2$ | $\cos^2(3)$ | $2x_{5}\dot{x}_{5}$ | $-2\sin(3)\cos(3)$ | 
| $x_{7}$ | $x_4\cdot x_{6}$ | $\frac{3}{2}\cos^2(3)$ | $x_4\dot{x}_{6}+\dot{x}_{4}x_6$ | $\frac{1}{2}\cdot \cos^2(3)-3\sin(3)\cos(3)$ |
| $x_{8}$ | $3x_1^3$ | $81$ | $9x_1^2\cdot \dot{x_1}$ | $81$ |
| $x_{9}$ | $x_3+x_7+x_8$ | $\sin(9)+\frac{3}{2}\cos^2(3)+81$ | $\dot{x}_3+\dot{x}_7+\dot{x}_8$ | $6\cos(9)+\frac{1}{2}\cos^2(3)-3\sin(3)\cos(3)+81$ |

Dual numbers augment the algebra of real numbers by adding a new component to each number. That new component, the epsilon value, is the derivative of the function at that number. Using dual numbers allows us to store both the function and it's derivative at any point. Thus, it is extremely useful to use dual number in automatic differentiation. We will use dual numbers to create an automatic differentiation package for Python.

  
## Software organization
Autodiff is organized as follows:
    ```
cs207-FinalProject/
             README.md (The current user's guide)
			 various hooks (travis.yml and so on)
             docs/  
                  READMEs from various milestones
             code/
                  autodiff.py (The key class and functions)
		     tests/
				various unit an integration tests
```
* Currently, you can install our package from GitHub.

#### Modules
Autoiff has just one module: autodiff.py. It contains the DualNumber class and all its accesories.

#### Tests
Autodiff is tested via `pytest`. To run the tests, navigate to TODO and run `TODO`. You can run an individual set of tests via `TODO`. 

#### Installation
Support for installation via `pip` is coming soon. 

In the mean time, follow the steps below:
1. clone or download autodiff from [github](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject)
2. in python files where you want to use autodiff, include the code ```import sys
sys.path.insert(0, '/path_to_autodiff/')
import autodiff as ad```

That's it! Note that editing sys.path is undone when you close python so removing autodiff from your system just requires you to delete the autodiff folder.

	  
## Implementation details
The autodiff package is dead simple: there is a single module (`autodiff`) and a single class (`DualNumber`), along with autodiff package versions of functions in the `math` module, e.g `sin` and `log`. It has no external dependencies.

Any `DualNumber` has two components: a value and a dictionary of derivatives. The value is the real-number result of whatever computation returned this dual number. The derivatives are a dictionary mapping variable names to real numbers, for instance `{'x':3, 'y':0.2}`. This would mean that the computation that produced this dual number depends on original inputs named x and y (and no others) and the derivative in the x direction is 3, while the derivative in the y direction is 1/5. Importantly, dual numbers don't care how they were produced, and can be the result of arbitrarially complex user-defined functions. In fact, (soon) any function that is written in pure python can simply be called on `DualNumber` inputs to get the derivatives at those input values.

Dual numbers work by simply updating the present derivatives in each direction at the same time a new value is computed. For example, the product rule: $\nabla xy = x\nabla y + y\nabla x$ says "to make the output's derivatives, take the derivatives stored in y and multiply them by x's value, then add the derivatives stored in x multiplied by y's value". 

#### Extensions
The following are coming to autodiff very sooon.
 - support comparison operators > and <
 - support DualNumber('x', np.ones(20)), where a single variable can refer to an array of values
 - streamline functions like ad.sin() to return a scalar when given scalar input
 - simplify handling of rsub, rpow and others
 - verify that our choice of branch in inverse trig functions matches `math` and `numpy`
 - add default arguments to log and exp
 - support non-differentiable functions like absolute value
 