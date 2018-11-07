[![Build Status](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject)

[![Coverage Status](https://coveralls.io/repos/github/cs207-project-erin-bruce-will/cs207-FinalProject/badge.svg)](https://coveralls.io/github/cs207-project-erin-bruce-will/cs207-FinalProject)

Erin, Bruce, and Will

# Milestone 2 Update



### How to Use *PackageName*

```python
import autodiff as ad

b = ad.DualNumber(None,2,{'x':1.2, 'y':9.5, 'z':5})
	
y = ad.cos(x)

y.value
-0.416146837

y.derivatives['x']
-1.09115691

y.derivatives['y']
-8.63832555

y.derivatives['z']
-4.54648713

```
Autodif works by defining dual numbers and then computing as usual. Built-in python operations are handled seamlessley, and operations imported from the math package (e.g. math.sin) must be replaced with thier Autodif equivalents (e.g ad.sin in the example above). Any function that is passed dual numbers as input will return an Autodif DualNumber object that stores the function's value and derivatives at the given point. Scalars are automatically promoted to DualNumbers as needed.

In milestone 2 we intend to deliver a parser that will assist novice users, either by upgrading exsiting code to work with autograd, or helping users write autograd-ready functions via a GUI.

### Introduction
Automatic differentiation is a set of techniques that allows a computer program to evaluate the derivative of a function as it evaluates the function's value. Automatic differentiation is a powerful tool that scales well with dimensonality of the inputs and outputs, and does not suffer from round off errors. When working with complex function of many inputs, manual calculation is unrealistic, and numeric differentiation requires a great deal of care and complex code. Automatic differentiation is, in fact, automatic, allowing the user to process functions quickly and efficiently.

### Background
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

### Software Organization

The following will be our preliminary directory structure:
    ```
cs207-FinalProject/
             README.md
	     .travis.yml
	     setup.cfg
             docs/  
                  milestone1
                  milestone2
                  milestone3
		  README_v1.md
                  technical documentation
                  user's guide
                  ...
             code/
                  autodiff.py
		  integration_test.py
		  test_binary_functions.py
	          test_unary_functions.py
		  write_unary_tests.py
                  separ.py
                  
             ...
```
* Currently, you can install our package from GitHub.
