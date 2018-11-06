[![Build Status](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject)

[![Coverage Status](https://coveralls.io/repos/github/cs207-project-erin-bruce-will/cs207-FinalProject/badge.svg)](https://coveralls.io/github/cs207-project-erin-bruce-will/cs207-FinalProject)

Erin, Bruce, and Will

# Milestone 2 Update

### Introduction
Automatic differentiation is a set of techniques that allows a computer program to evaluate the derivative of a function as it evaluates the function's value. Automatic differentiation is a powerful tool that scales well with dimensonality of the inputs and outputs, and does not suffer from round off errors. When working with complex function of many inputs, manual calculation is unrealistic, and numeric differentiation requires a great deal of care and complex code. Automatic differentiation is automatic, allowing 

### Background
Automatic differentiation (AD) relies on the chain rule to perform elementary derivatives at the same time it performs the elementary operations which make up the function. Elementary arithemetic operations (such as add, subtract, multiply, divide, etc.) and elementary functions (such as exp, sine, cosine, etc.) are performed at each node, and AD provides the instruction to take the derivate of the given elementary operation or function. These derivatives are accumulated via the chain rule to return the full derivative of a function. There are two modes employed: forward mode, which performs differentiation based on the independent or predictor variable, and reverse mode, which performs differentiation based on the dependent or response variable.

### How to Use *PackageName*

```python
import autodiff as ad

b = DualNumber(None,-2/3,{'x':1.2, 'y':9.5, 'z':5})
	
y = cos(x)

y.value
y.derivatives['x','y','z']

a = ad.DualNumber('a',7)
b = ad.DualNumber('b',8)

val_at_7_8, grad_at_7_8 = myfun(a,b).eval()

a = ad.DualNumber('a',1)
b = ad.DualNumber('b',2)
val_at_1_2, grad_at_1_2 = myfun(a,b).eval()

val2_at_1_2, grad2_at_1_2 = myfun2(a,b).eval()
jacobian_at_1_2 = [grad_at_1_2, grad2_at_1_2]
```
Autodif works by defining dual numbers and then computing as usual. Built-in python operations are handled seamlessley, and operations imported from the math package (e.g. math.sin) must be replaced with thier Autodif equivalents (e.g ad.sin in the example above). Any function that is passed dual numbers as input will return an Autodif DualNumber object that stores the function's value and derivatives at the given point. Scalars are automatically promoted to DualNumbers as needed.

In milestone 2 we intend to deliver a parser that will assist novice users, either by upgrading exsiting code to work with autograd, or helping users write autograd-ready functions via a GUI.


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
	

User wants to evaluate derivatives at $a=3$, and provide a functional form $f(x)=sin(x^2)+\frac{1}{2}x\cdot cos^2(x)+3x^3$. Then inside our software, an automatic parser `separ` decomposes the function into pieces according to the add/minus sign:

$$f_1,f_2,f_3 = separ(f_x)$$
    
For $f_1, f_2, f_3$, a dual number object is created:

$$x = myAD(a)$$

For each piece (i.e. $f_1,f_2,f_3$, etc.), second parser `myParser` create layers. For example, the second piece $f_2$ should be further parsed as $y_1 = \frac{1}{2}x_1$, $y_2 = \cos(x_1)$, and $y_3 = y_2^2$:

$$y1,y2,y3 = myParser(f2)$$

Now evaluate, for instance the functional value and derivative for piece $f_2$, where $x$ is a `myAD` object:
$$f_2=\frac{1}{2}x cos^2(x)$$

**----------------------------------------------End of Bruce Edit-------------------------------------------**<BR>


### Software Organization
Discuss how you plan on organizing your software package.<BR>
* What will the directory structure look like?
    - The following will be our preliminary directory structure:
    ```
project_repo/
             README.md
             docs/  
                  milestone1
                  milestone2
                  milestone3
                  technical documentation
                  user's guide
                  ...
             package_deliverable/
                  autodif.py
                  separ.py
                  ...
             test/
                  
             ...
```
* What modules do you plan on including? What is their basic functionality?
    - Based on the prior section, we plan on delivering the following modules:
		* `autodif`: Implements the DualNumber class and the extra elementary functions (`ad.sin`, `ad.exp`, and so on)
        * `separ`: a preliminary parsing class to break down user's formatted inputs into pieces. We can separate by addition/subtraction or some other basic operation (e.g. log, exp, etc.)
    - In addition, we may include some GUI modules and/or I/O modules - more details will follow in milestone 2 document.
* Where will your test suite live? Will you use TravisCI? Coveralls?
    - We will store all our tests in the *test* folder. We will create virtual environment for individual test.
    - Yes, we've set up TravisCI and Coveralls for our project.

* How will you distribute your package (e.g. PyPI)
    - Yes, ...
    - 
    

### Implementation
Discuss how you plan on implementing the forward mode of automatic differentiation.
* What are the core data structures?
	- Our funadamental object will be a DualNumber. These objects will have a .value field and a .grad field. The former will store a real number, ultimately the numerical result of user-defined computations. The latter will hold a dictionary of the 'epsilon' components, e.g. {'a':4, 'b': 5.5}, encoding that the derivative w.r.t. a at the given point is 4, the derivative w.r.t b is 5.5, and no other inputs affect the value of the function; their derivatives are zero.
	- Computing with dual numbers is striaghtforward: all dual numbers store derivatives with respect to given directions, and oprations on dual numbers simply need to produce the correct value and correct dictionary list. Under the hood, we perform algebra as if there is $\epsilon_a$ and $epsilon_b$ (and so on) under the rule that the product of any two espilons is zero. We have verified that this gives correct gradients under the product and sum rules, so should hold generally.
* What classes will you implement?
	- We only need to implement the DualNumber class and overload its operators and python's 'math' package. Everything else follows naturally.
* What method and name attributes will your classes have?
	- DualNumbers will have `.val` to access the function's value, `.grad` to access the gradient, and `.eval` to access both.
* What external dependencies will you rely on?
	- We do not believe we require any external dependencies beyond python's built-in `math` module. If we upgrade DualNumber to store a vector of values instead of scalars, we will need numpy. This would purely be a convienience to users, allowing them to write sin(vector) instead of taking each sin() individually. The power of the library is the same under both modes.
* How will you deal with elementary functions like `sin` and `exp`?
	- We will require users to make use of ad.sin and ad.exp, in the same way numpy requires users to work with np.sin() and np.exp(). As a special case, we'll have to handle $a^x$ $x^a$ and $x^x$ differently ($a$ a constant and $x$ a variable).(Unless we opt to implemnt only the most general case ($x^y$) for x and y both dual numbers) For example, when we implement the derivative of function $f(x)=x^x$, we should take consideration both the form of $x^a$ and the form of $a^x$ and obtain the derivative by chain rule $\left(e.g. \left(x^a\right)'=ax^{a-1}\cdot x'+x^aln(x)\cdot (a)'=ax^{a-1}+0=ax^{a-1}\right)$
