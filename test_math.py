import pytest

#TODO: logs, exponents, and unitary functions

@pytest.fixture
def a():
	from autodiff import DualNumber
	a = DualNumber(None,3.5,{'y':4,'x':3})
	return a

@pytest.fixture
def b():
	from autodiff import DualNumber
	b = DualNumber(None,2,{'x':1.2, 'y':9.5, 'z':5})
	return b

@pytest.fixture
def s():
	return 4.2

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
	assert output.derivatives == {'x':3/s, 'y':4/s}
	
def test_div_sd(s,b):
	output = s/b
	assert output.value == b.value/s
	coef = -s*b.value**2
	assert output.derivatives == {'x':coef*1.2, 'y':coef*9.5, 'z':coef*5}
	
def test_div_dd(a,b):
	output = a/b
	assert output.value == a.value+b.value
	den = b.value**2
	f = a.value
	g = b.value
	assert output.derivatives == {'x':(g*3-f*1.2)/den, 'y':(g*4-f*9.5)/den, 'z':(g*0-f*5)/den}

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
# scalar ^ dual
# dual ^ scalar
# dual ^ dual

# formula: del f^g = f^g(g/f del f + ln(f)del g)
def test_exponents(a,b):
	pass

###
# logs
###
# log_dual(scalar)
# log_scalar(dual)
# log_dual(dual)
def test_log(a,b):
	pass


#########	
# single input
#########

# negtion
# f(dual)
def test_neg(a,b):
	pass
	
# plan: write function to build needed file
# should write a few until I get bored
#(f, f`)
#(sin,cos)	
	
	
	
	
def test_sin(a,b):
	pass
	
def test_cos(a,b):
	pass
	
def test_tan(a,b):
	pass
	
def test_cot(a,b):
	pass
	
def test_sec(a,b):
	pass
	
def test_csc(a,b):
	pass
	
def test_arcsin(a,b):
	pass

def test_arccos(a,b):
	pass
	
def test_arctan(a,b):
	pass
	
def test_arcsec(a,b):
	pass
	
def test_arccsc(a,b):
	pass