# write funciton name and 

# name, value computation, derivative computation, argument
test_tuples = [
    ("sin",        "math.sin(x)",         "math.cos(x)",                             ""),
    ("cos",     "math.cos(x)",         "-math.sin(x)",                         ""),
    ("tan",     "math.tan(x)",         "1/(math.cos(x)**2)",                     ""),
    ("cot",     "1/math.tan(x)",     "-1/(math.sin(x)**2)",                     ""),
    ("sec",     "1/math.cos(x)",     "1/math.cos(x)*math.tan(x)",             ""),
    ("csc",     "1/math.sin(x)",     "-1/math.sin(x)*1/math.tan(x)",         ""),
    ("arcsin",     "math.asin(x)",     "1/math.sqrt(1-x**2)",                     ""),
    ("arccos",     "math.acos(x)",     "-1/math.sqrt(1-x**2)",                 ""),
    ("arctan",     "math.atan(x)",     "1/(1+x**2)",                             ""),
    ("arccot",     "math.pi/2 - math.atan(x)",     "-1/(1+x**2)",                     ""),
    ("arcsec",     "math.acos(1/x)",     "1/(abs(x)*math.sqrt(x**2-1))",     "inv"),
    ("arccsc",     "math.asin(1/x)",     "-1/(abs(x)*math.sqrt(x**2-1))",     "inv")
]

preamble = """import pytest
import autodiff as ad
import math
import numpy as np

@pytest.fixture
def b():
    from autodiff import DualNumber
    b = DualNumber._from_dict(-2/3,{'x':1.2, 'y':9.5, 'z':5})
    return b
    
@pytest.fixture
def binv():
    from autodiff import DualNumber
    binv = DualNumber._from_dict(-3/2,{'x':1.2, 'y':9.5, 'z':5})
    return binv

@pytest.fixture
def s():
    return .42
    
@pytest.fixture
def sinv():
    return 1/.42
    
@pytest.fixture
def m():
    m = np.array([1/6,2/6,3/6])
    return m
    
@pytest.fixture
def minv():
    minv = 1/np.array([1/6,2/6,3/6])
    return minv
    
@pytest.fixture
def dm():
    from autodiff import DualNumber
    dm = DualNumber._from_dict(np.array([1/6,2/6,3/6]),{'y':np.array([4,-5,6]),'x':np.array([8,7,-9])})
    return dm
    
@pytest.fixture
def dminv():
    from autodiff import DualNumber
    dminv = DualNumber._from_dict(1/np.array([1/6,2/6,3/6]),{'y':np.array([4,-5,6]),'x':np.array([8,7,-9])})
    return dminv
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
def test_{0}_s(s{3}):
    def value(x):
        return {1}
    output = ad.{0}(s{3})
    assert output.value == pytest.approx(value(s{3}))
    assert output.derivatives == {{}}

def test_{0}_d(b{3}):
    def value(x):
        return {1}
    def der(x):
        return {2}
    
    output = ad.{0}(b{3})
    assert output.value == pytest.approx(value(b{3}.value))
    coef = der(b{3}.value)
    assert output.derivatives == pytest.approx({{'x':coef*1.2, 'y':coef*9.5, 'z':coef*5}})
    
def test_{0}_no_change(b{3}):
    b_d = b{3}.derivatives
    output = ad.{0}(b{3})
    assert b_d == b{3}.derivatives
    
def test_{0}_sm(m{3}):
    def value(x):
        return {1}
    output = ad.{0}(m{3})
    for i in range(len(m{3})):
        assert output.value[i] == pytest.approx(value(m{3}[i]))
    assert output.derivatives == {{}}
        
def test_{0}_dm(dm{3}):
    def value(x):
        return {1}
    def der(x):
        return {2}
   
    output = ad.{0}(dm{3})
    for i in range(len(dm{3}.value)):
        assert output.value[i] == pytest.approx(value(dm{3}.value[i]))
        coef = der(dm{3}.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm{3}.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm{3}.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm{3}.derivatives['z'][i])

    
"""

with open("tests/test_unary_functions.py",'w') as outfile:
    outfile.write(preamble)
    outfile.write(negation_test)
    for cur_tup in test_tuples:
        outfile.write(test_pattern.format(cur_tup[0],cur_tup[1], cur_tup[2], cur_tup[3]))
