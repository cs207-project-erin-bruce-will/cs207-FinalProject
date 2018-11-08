import pytest
import ourProject.autodiff as ad
import math
#TODO: logs, exponents, and unitary functions

@pytest.fixture
def a():
    from ourProject.autodiff import DualNumber
    a = DualNumber(None,3.5,{'y':4,'x':3})
    return a

@pytest.fixture
def b():
    from ourProject.autodiff import DualNumber
    b = DualNumber(None,2,{'x':1.2, 'y':9.5, 'z':5})
    return b

@pytest.fixture
def s():
    return 4.2
	
@pytest.fixture
def s2():
    return 3.1

###
# Addition
###
def test_add_ds(a,s):
    output = a+s
    assert output.value == a.value+s
    assert output.derivatives == {'x':3, 'y':4}
    
def test_add_sd(s,b):
    output = s+b
    assert output.value == b.value+s
    assert output.derivatives == {'x':1.2, 'y':9.5, 'z':5}
    
def test_add_dd(a,b):
    output = a+b
    assert output.value == a.value+b.value
    assert output.derivatives == {'x':1.2+3, 'y':9.5+4, 'z':5}

def test_add_no_change(a,b):
    a_d = a.derivatives
    b_d = b.derivatives
    output = a+b
    assert a_d == a.derivatives
    assert b_d == b.derivatives

###
# Subtraction
###
def test_sub_ds(a,s):
    output = a-s
    assert output.value == a.value-s
    assert output.derivatives == {'x':3, 'y':4}
    
def test_sub_sd(s,b):
    output = s-b
    assert output.value == s-b.value
    assert output.derivatives == {'x':-1.2, 'y':-9.5, 'z':-5}
    
def test_sub_dd(a,b):
    output = a-b
    assert output.value == a.value-b.value
    assert output.derivatives == {'x':3-1.2, 'y':4-9.5, 'z':-5}

def test_sub_no_change(a,b):
    a_d = a.derivatives
    b_d = b.derivatives
    output = a-b
    assert a_d == a.derivatives
    assert b_d == b.derivatives
    output = b-a
    assert a_d == a.derivatives
    assert b_d == b.derivatives

###
# Multiplication
###
def test_mul_ds(a,s):
    output = a*s
    assert output.value == a.value*s
    assert output.derivatives == {'x':s*3, 'y':s*4}
    
def test_mul_sd(s,b):
    output = s*b
    assert output.value == b.value*s
    assert output.derivatives == {'x':1.2*s, 'y':9.5*s, 'z':5*s}
    
def test_mul_dd(a,b):
    output = a*b
    assert output.value == a.value*b.value
    assert output.derivatives == {'x':3.5*1.2+2*3 , 'y':3.5*9.5+2*4 , 'z':3.5*5}

def test_mul_no_change(a,b):
    a_d = a.derivatives
    b_d = b.derivatives
    output = a*b
    assert a_d == a.derivatives
    assert b_d == b.derivatives
    
###
# Division
###
def test_div_ds(a,s):
    output = a/s
    assert output.value == a.value/s
    assert output.derivatives == pytest.approx({'x':3/s, 'y':4/s})
    
def test_div_sd(s,b):
    output = s/b
    assert output.value == s/b.value
    coef = -s/(b.value**2)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_div_dd(a,b):
    output = a/b
    assert output.value == a.value/b.value
    den = b.value**2
    f = a.value
    g = b.value
    assert output.derivatives == pytest.approx({'x':(g*3-f*1.2)/den, 'y':(g*4-f*9.5)/den, 'z':(g*0-f*5)/den})

def test_div_no_change(a,b):
    a_d = a.derivatives
    b_d = b.derivatives
    output = a/b
    assert a_d == a.derivatives
    assert b_d == b.derivatives
    output = b/a
    assert a_d == a.derivatives
    assert b_d == b.derivatives

###
# Exponents
###
# formula: del f^g = f^g(g/f del f + ln(f)del g)
def test_exponents_ds(a,s):
    output = a**s
    assert output.value == a.value**s
    m = s*a.value**(s-1)
    assert output.derivatives == {'x':m*3, 'y':m*4}
    
def test_exponents_sd(s,b):
    output = s**b
    assert output.value == s**b.value
    m = s**b.value*math.log(s)
    assert output.derivatives == {'x':m*1.2, 'y':m*9.5, 'z':m*5}
    
def test_exponents_dd(a,b):
    output = a**b
    assert output.value == a.value**b.value
    m  = a.value**b.value
    ma = b.value/a.value
    mb = math.log(a.value)
    assert output.derivatives == {'x':m*ma*3+m*mb*1.2, 'y':m*ma*4+m*mb*9.5, 'z':m*mb*5}

def test_exponents_change(a,b):
    a_d = a.derivatives
    b_d = b.derivatives
    output = a**b
    assert a_d == a.derivatives
    assert b_d == b.derivatives
###
# logs
###
# log_f(g)
# formula: log_b(a) = log(b)/alog(b)**2 del a + -log(a)/blog(a)**2 del b
def test_log_ss(s,s2):
    output = ad.log(s,s2)
    assert output.value == math.log(s, s2)
    assert output.derivatives == {}

def test_log_ds(a,s):
    output = ad.log(a,s)
    assert output.value == math.log(a.value, s)
    m  = 1/math.log(s)**2
    ma = math.log(s)/a.value
    assert output.derivatives == pytest.approx({'x':m*ma*3, 'y':m*ma*4})
    
def test_log_sd(s,b):
    output = ad.log(s,b)
    assert output.value == math.log(s, b.value)
    m  = 1/math.log(b.value)**2
    mb = -math.log(s)/b.value
    assert output.derivatives == pytest.approx({'x':m*mb*1.2, 'y':m*mb*9.5, 'z':m*mb*5})

def test_log_dd(a,b):
    output = ad.log(a,b)
    assert output.value == math.log(a.value,b.value)
    m  = 1/math.log(b.value)**2
    ma = math.log(b.value)/a.value
    mb = -math.log(a.value)/b.value
    assert output.derivatives == pytest.approx({'x':m*ma*3+m*mb*1.2, 'y':m*ma*4+m*mb*9.5, 'z':m*mb*5})

def test_log_change(a,b):
    a_d = a.derivatives
    b_d = b.derivatives
    output = ad.log(a,b)
    assert a_d == a.derivatives
    assert b_d == b.derivatives
