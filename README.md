[![Build Status](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/cs207-project-erin-bruce-will/cs207-FinalProject)

[![Coverage Status](https://coveralls.io/repos/github/cs207-project-erin-bruce-will/cs207-FinalProject/badge.svg)](https://coveralls.io/github/cs207-project-erin-bruce-will/cs207-FinalProject)

# AutoDiff

Developed by: Will Claybaugh, Bruce Xiong, Erin Williams  
Group #3, CS207 Fall 2018


## Introduction
Autodiff finds the derivatives of a function (to machine precision!) at the same time it finds the value of the function.
```
import autodiff as ad

x = ad.DualNumber('x', 2)
y = ad.DualNumber('y', 3)

out = x/y
out.value # 0.66666, the value of 2 divided by 3
out.derivatives #{x: 1/3, y: -2/(3**2)}, the gradient of x/y at (2,3)
```
Autodiff works for functions and expressions with any number of inputs. Just pass those functions DualNumbers instead of regular ints/floats (and upgrade any math module functions to their autodiff equvalents)

## Installation
Autodiff can be installed using ```pip install AutoDiff-group3```.

Autodiff can also be installed by downloading from [github](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject). Becuase it has no dependencies, you can simply add the repo folder to your python path (```import sys
sys.path.insert(0, '/path_to_repo/')```) and import as normal. 

## Examples

Using autodiff is very simple:
```python
import sys
sys.path.insert(0, "C:/Users/erina/cs207-FinalProject/")
import autodiff as ad

def f(a,b):
    return 3*a/b*ad.sin(a*b+2)

out = f(ad.DualNumber('x',2),ad.DualNumber('y',3))

print(out.value)
1.978716

print(out.derivatives['x'])
0.116358

print(out.derivatives['y'])
-1.24157

# get the value and derifative of f at a different point
out = f(ad.DualNumber('x',0),ad.DualNumber('y',1))
```

A Python 3 notebook containing more in-depth examples and usage is available [HERE](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject/blob/master/docs/Demo.ipynb)

## Documentation

Click [HERE](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject/blob/master/docs/documentation.md) for full documentation.

## Dependencies

Click [HERE](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject/blob/master/docs/requirements.txt) for a full listing of dependencies.

## License

Click [HERE](https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject/blob/master/LICENSE) to view our MIT License.

 
