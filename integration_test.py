import pytest
import autodiff
import math

#TODO: test things like x**(-1) == 1/x
#TODO: complex functions involving multiple variables

@pytest.fixture
def x():
	return autodiff.DualNumber('a',2)

b = 5.8
c = 3.0
d = -2.75
e = 4

def test1(x):
	output = e*x**b + c
	assert output.value == -219.86094420380775
	assert output.derivative['x'] == -646.2967381910424
	
def test2(x):
	output = d*autoDiff.exp(b,x) * x**c-autoDiff.cos(x)/x
	assert output.value == -739.8719265817265
	assert output.derivative['x'] == -2410.7248756178847
	
def test3(x):
	output = (c*x + autoDiff.logx(x,b))**(x**(1/2)+1/x)
	assert output.value == 60.621261109269476
	assert output.derivative['x'] == 29.37478740554978

def test4(x):
	output = x-autoDiff.exp(-2*autoDiff.sin(4*x)**2)
	assert output.value == 1.858811511058501
	assert output.derivative['x'] == 0.6748109260705084

def test5(x):
	output = autoDiff.exp(-(x+autoDiff.cos(3*x)**2)**0.5)*autoDiff.sin(x*autoDiff.log(1+x**2))
	assert output.value == -0.013972848911640818
	assert output.derivative['x'] == -0.5684464955717082

def test6(x):
	output = autoDiff.log(3*x)/(x**x)
	assert output.value == 0.44793986730701374
	assert output.derivative['x'] == -0.6334281233912664

def test7(x):
	output = autoDiff.logx(x**(1/3),autoDiff.arccos(autoDiff.exp(-3/x)))
	assert output.value == 1.2853017513220446
	assert output.derivative['x'] == -1.47926913780827
	
