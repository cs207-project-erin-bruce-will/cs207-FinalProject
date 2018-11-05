import pytest
import autodiff as ad
import math

#TODO: test things like x**(-1) == 1/x
#TODO: complex functions involving multiple variables

@pytest.fixture
def x():
    import autodiff
    return autodiff.DualNumber('x',2)

b = 5.8
c = 3.0
d = -2.75
e = 4

def test0(x):
    output = e*x + c
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
    output = (c*x + ad.log(x,b))**(x**(1/2)+1/x)
    assert output.value == 60.621261109269476
    assert output.derivatives['x'] == 29.37478740554978

def test4(x):
    output = x-math.e**(-2*ad.sin(4*x)**2)
    assert output.value == 1.858811511058501
    assert output.derivatives['x'] == 0.6748109260705084

def test5(x):
    output = math.e**(-(x+ad.cos(3*x)**2)**0.5)*ad.sin(x*ad.log(math.e,1+x**2))
    assert output.value == -0.013972848911640818
    assert output.derivatives['x'] == -0.5684464955717082

def test6(x):
    output = ad.log(math.e,3*x)/(x**x)
    assert output.value == 0.44793986730701374
    assert output.derivatives['x'] == -0.6334281233912664

def test7(x):
    output = ad.log(x**(1/3),ad.arccos(math.e**(-3/x)))
    assert output.value == 1.2853017513220446
    assert output.derivatives['x'] == -1.47926913780827
    
