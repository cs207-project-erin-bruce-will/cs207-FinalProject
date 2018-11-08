import pytest
import ourProject.autodiff as ad
import math

#TODO: test things like x**(-1) == 1/x
#TODO: complex functions involving multiple variables

@pytest.fixture
def x():
	from ourProject.autodiff import DualNumber
	return DualNumber('x',2)

@pytest.fixture
def y():
	from ourProject.autodiff import DualNumber
	y = DualNumber('y',3.5)
	return y

	
b = 5.8
c = 3.0
d = -2.75
e = 4

def test0(x):
	output = e*x+c
	assert output.value == 4*2+3
	assert output.derivatives['x'] == 4

def test1(x):
	output = e*(x**b) + c
	assert output.value == 4*x.value**(5.8)+3
	assert output.derivatives['x'] == pytest.approx(4*5.8*(x.value**(5.8-1)))
	
def test2(x):
	output = d*(b**x) * x**c-ad.cos(x)/x
	assert output.value == -739.8719265817265
	assert output.derivatives['x'] == -2410.7248756178847
	
def test3(x):
	output = (c*x + ad.log(b,x))**(x**(1/2)+1/x)
	assert output.value == 60.621261109269476
	assert output.derivatives['x'] == 29.37478740554978

def test4(x):
	output = x-math.e**(-2*ad.sin(4*x)**2)
	assert output.value == 1.858811511058501
	assert output.derivatives['x'] == 0.6748109260705084

def test5(x):
	output = math.e**(-(x+ad.cos(3*x)**2)**0.5)*ad.sin(x*ad.log(1+x**2,math.e))
	assert output.value == -0.013972848911640818
	assert output.derivatives['x'] == -0.5684464955717082

def test6(x):
	output = ad.log(3*x,math.e)/(x**x)
	assert output.value == 0.44793986730701374
	assert output.derivatives['x'] == -0.6334281233912664

def test7(x):
	output = ad.log(ad.arccos(ad.exp(-3/x)),x**(1/3))
	assert output.value == 1.2853017513220446
	assert output.derivatives['x'] == -1.47926913780827
	

def test_2d(x,y):
	def f(a,b):
		return (a/b)*ad.sin(a*b)
	delta = .0000000001
	output = f(x,y)
	dx = (f(x.value+delta,y.value)-f(x.value,y.value))/delta
	dy = (f(x.value,y.value+delta)-f(x.value,y.value))/delta
	assert output.derivatives['x'] == pytest.approx(dx.value)
	assert output.derivatives['y'] == pytest.approx(dy.value)
