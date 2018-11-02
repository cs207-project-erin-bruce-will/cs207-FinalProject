# write funciton name and 
test_tuples = [
	("sin", "math.cos(x)"),
	("cos", "-math.sin(x)"),
	("tan", "math.sec(x)**2"),
	("cot", "-math.csc(x)**2"),
	("sec", "math.sec(x)*math.tan(x)"),
	("csc", "-math.csc(x)*math.cot(x)"),
	("arcsin", "1/math.sqrt(1-x**2)"),
	("arccos", "-1/math.sqrt(1-x**2)"),
	("arctan", "1/(1+x**2)"),
	("arccot", "-1/(1+x**2)"),
	("arcsec", "1/(math.abs(x)*math.sqrt(x**2-1))"),
	("arccsc", "-1/(math.abs(x)*math.sqrt(x**2-1))")
]

preamble = """import pytest
import autodiff as ad

@pytest.fixture
def b():
    from autodiff import DualNumber
    b = DualNumber(None,-2/3,{'x':1.2, 'y':9.5, 'z':5})
    return b

@pytest.fixture
def s():
    return .42
"""

negation_test = """
###
# negation
###

def test_negation_d(b):
	output = -b
	assert output.value == -b.value
	assert output.derivatives == {'x':-1.2, 'y':-9.5, 'z':-5}
	
def test_negation_no_change(b):
    b_d = b.derivatives
    output = -b
    assert b_d == b.derivatives

"""

test_pattern = """
###
# {0}
###
def test_{0}_s(s):
	output = ad.{0}(s)
	assert output.value == math.{0}(s)
	assert output.derivatives == {{'x':0,'y':0,'z':0}}

def test_{0}_d(b):
	def der(x):
		return {1}
	
	output = ad.{0}(b)
	assert output.value == math.{0}(b.value)
	assert output.derivatives == {{'x':der(1.2), 'y':der(9.5), 'z':der(5)}}
	
def test_{0}_no_change(b):
    b_d = b.derivatives
    output = ad.{0}(b)
    assert b_d == b.derivatives
	
"""

with open("test_unary_functions.py",'w') as outfile:
	outfile.write(preamble)
	outfile.write(negation_test)
	for cur_tup in test_tuples:
		outfile.write(test_pattern.format(cur_tup[0],cur_tup[1]))
