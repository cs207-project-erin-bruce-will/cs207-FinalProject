###
# Creation
###
import autodiff as ad
import numpy as np
import math
import pytest

# name, number
def test_creation_num():
    x = ad.DualNumber('x',10)
    assert x.value == 10
    assert x.derivatives == {'x':1}

# name, array
def test_creation_arr():
    x = ad.DualNumber('x',np.array([10,12]))
    assert np.all(x.value == np.array([10,12]))
    assert np.all(x.derivatives['x[0]'] == np.array([1,0]))
    assert np.all(x.derivatives['x[1]'] == np.array([0,1]))

# missing inputs
def test_missing_value():
    with pytest.raises(TypeError):
        ad.DualNumber('x')

# bad inputs
def test_bad_value():
    with pytest.raises(TypeError):
        ad.DualNumber('x',"yes")

def test_bad_name():
    with pytest.raises(TypeError):
        ad.DualNumber(7,20)

# value is already a dual number
def test_dual_value():
    z = ad.DualNumber('z',10)
    with pytest.raises(TypeError):
        ad.DualNumber('x',z)


###
# Promotion
###

# promote numeric
def test_promote_n():
    x = ad.DualNumber.promote(7)
    assert x.value==7
    assert x.derivatives == {}
    
# promote array
def test_promote_arr():
    x = ad.DualNumber.promote(np.array([[7,3],[2,1]]))
    assert np.all(x.value==np.array([[7,3],[2,1]]))
    assert x.derivatives == {}

# promote dual
def test_promote_dualarr():
    x = ad.DualNumber.promote(np.array([[7,3],[2,1]]))
    z = ad.DualNumber.promote(x)
    assert np.all(z.value==np.array([[7,3],[2,1]]))
    assert x.derivatives == {}

# promote other
def test_promote_other():
    with pytest.raises(TypeError):
        ad.DualNumber.promote("yes")

###
# Corner cases
###
        
# x**1/3 has vertical tangent at 0
def test_vert_tan():
    x = ad.DualNumber('x',0)
    output = x**(1/3)
    assert output.value == 0
    assert output.derivatives['x'] == float("inf")
    assert output.derivatives['x'] > 0
    

def test_undefined_value():
    x = ad.DualNumber('x',0)
    with pytest.raises(ZeroDivisionError):
        1/x

# sin(1/x) goes nuts near zero
def test_oscilating():
    x = ad.DualNumber('x',0.001)
    output = ad.sin(1/x)
    assert output.value == math.sin(1/0.001)
    assert output.derivatives['x'] == math.cos(1/.001)*(-1/(.001**2))

# sin(1/x)/x goes nuts, but closes in on zero
def test_oscilating2():
    x = ad.DualNumber('x',0.001)
    output = x*ad.sin(1/x)
    assert output.value == .001*math.sin(1/0.001)
    assert pytest.approx(output.derivatives['x'] == math.sin(1/.001) - math.cos(1/.001)/.001)


# branch functions (different left and right halves)
def test_branch():
    def my_fun(x):
        if x<10:
            return x
        else:
            return x**2
    x = ad.DualNumber('x',10)
    with pytest.warns(UserWarning):
        my_fun(x)
    
    x2 = ad.DualNumber('x',20)
    my_fun(x2)

