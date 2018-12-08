import pytest
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


###
# sin
###
def test_sin_s(s):
    def value(x):
        return math.sin(x)
    output = ad.sin(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_sin_d(b):
    def value(x):
        return math.sin(x)
    def der(x):
        return math.cos(x)
    
    output = ad.sin(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_sin_no_change(b):
    b_d = b.derivatives
    output = ad.sin(b)
    assert b_d == b.derivatives
    
def test_sin_sm(m):
    def value(x):
        return math.sin(x)
    output = ad.sin(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_sin_dm(dm):
    def value(x):
        return math.sin(x)
    def der(x):
        return math.cos(x)
   
    output = ad.sin(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# cos
###
def test_cos_s(s):
    def value(x):
        return math.cos(x)
    output = ad.cos(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_cos_d(b):
    def value(x):
        return math.cos(x)
    def der(x):
        return -math.sin(x)
    
    output = ad.cos(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_cos_no_change(b):
    b_d = b.derivatives
    output = ad.cos(b)
    assert b_d == b.derivatives
    
def test_cos_sm(m):
    def value(x):
        return math.cos(x)
    output = ad.cos(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_cos_dm(dm):
    def value(x):
        return math.cos(x)
    def der(x):
        return -math.sin(x)
   
    output = ad.cos(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# tan
###
def test_tan_s(s):
    def value(x):
        return math.tan(x)
    output = ad.tan(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_tan_d(b):
    def value(x):
        return math.tan(x)
    def der(x):
        return 1/(math.cos(x)**2)
    
    output = ad.tan(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_tan_no_change(b):
    b_d = b.derivatives
    output = ad.tan(b)
    assert b_d == b.derivatives
    
def test_tan_sm(m):
    def value(x):
        return math.tan(x)
    output = ad.tan(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_tan_dm(dm):
    def value(x):
        return math.tan(x)
    def der(x):
        return 1/(math.cos(x)**2)
   
    output = ad.tan(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# cot
###
def test_cot_s(s):
    def value(x):
        return 1/math.tan(x)
    output = ad.cot(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_cot_d(b):
    def value(x):
        return 1/math.tan(x)
    def der(x):
        return -1/(math.sin(x)**2)
    
    output = ad.cot(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_cot_no_change(b):
    b_d = b.derivatives
    output = ad.cot(b)
    assert b_d == b.derivatives
    
def test_cot_sm(m):
    def value(x):
        return 1/math.tan(x)
    output = ad.cot(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_cot_dm(dm):
    def value(x):
        return 1/math.tan(x)
    def der(x):
        return -1/(math.sin(x)**2)
   
    output = ad.cot(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# sec
###
def test_sec_s(s):
    def value(x):
        return 1/math.cos(x)
    output = ad.sec(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_sec_d(b):
    def value(x):
        return 1/math.cos(x)
    def der(x):
        return 1/math.cos(x)*math.tan(x)
    
    output = ad.sec(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_sec_no_change(b):
    b_d = b.derivatives
    output = ad.sec(b)
    assert b_d == b.derivatives
    
def test_sec_sm(m):
    def value(x):
        return 1/math.cos(x)
    output = ad.sec(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_sec_dm(dm):
    def value(x):
        return 1/math.cos(x)
    def der(x):
        return 1/math.cos(x)*math.tan(x)
   
    output = ad.sec(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# csc
###
def test_csc_s(s):
    def value(x):
        return 1/math.sin(x)
    output = ad.csc(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_csc_d(b):
    def value(x):
        return 1/math.sin(x)
    def der(x):
        return -1/math.sin(x)*1/math.tan(x)
    
    output = ad.csc(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_csc_no_change(b):
    b_d = b.derivatives
    output = ad.csc(b)
    assert b_d == b.derivatives
    
def test_csc_sm(m):
    def value(x):
        return 1/math.sin(x)
    output = ad.csc(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_csc_dm(dm):
    def value(x):
        return 1/math.sin(x)
    def der(x):
        return -1/math.sin(x)*1/math.tan(x)
   
    output = ad.csc(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# arcsin
###
def test_arcsin_s(s):
    def value(x):
        return math.asin(x)
    output = ad.arcsin(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_arcsin_d(b):
    def value(x):
        return math.asin(x)
    def der(x):
        return 1/math.sqrt(1-x**2)
    
    output = ad.arcsin(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_arcsin_no_change(b):
    b_d = b.derivatives
    output = ad.arcsin(b)
    assert b_d == b.derivatives
    
def test_arcsin_sm(m):
    def value(x):
        return math.asin(x)
    output = ad.arcsin(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_arcsin_dm(dm):
    def value(x):
        return math.asin(x)
    def der(x):
        return 1/math.sqrt(1-x**2)
   
    output = ad.arcsin(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# arccos
###
def test_arccos_s(s):
    def value(x):
        return math.acos(x)
    output = ad.arccos(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_arccos_d(b):
    def value(x):
        return math.acos(x)
    def der(x):
        return -1/math.sqrt(1-x**2)
    
    output = ad.arccos(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_arccos_no_change(b):
    b_d = b.derivatives
    output = ad.arccos(b)
    assert b_d == b.derivatives
    
def test_arccos_sm(m):
    def value(x):
        return math.acos(x)
    output = ad.arccos(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_arccos_dm(dm):
    def value(x):
        return math.acos(x)
    def der(x):
        return -1/math.sqrt(1-x**2)
   
    output = ad.arccos(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# arctan
###
def test_arctan_s(s):
    def value(x):
        return math.atan(x)
    output = ad.arctan(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_arctan_d(b):
    def value(x):
        return math.atan(x)
    def der(x):
        return 1/(1+x**2)
    
    output = ad.arctan(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_arctan_no_change(b):
    b_d = b.derivatives
    output = ad.arctan(b)
    assert b_d == b.derivatives
    
def test_arctan_sm(m):
    def value(x):
        return math.atan(x)
    output = ad.arctan(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_arctan_dm(dm):
    def value(x):
        return math.atan(x)
    def der(x):
        return 1/(1+x**2)
   
    output = ad.arctan(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# arccot
###
def test_arccot_s(s):
    def value(x):
        return math.pi/2 - math.atan(x)
    output = ad.arccot(s)
    assert output.value == pytest.approx(value(s))
    assert output.derivatives == {}

def test_arccot_d(b):
    def value(x):
        return math.pi/2 - math.atan(x)
    def der(x):
        return -1/(1+x**2)
    
    output = ad.arccot(b)
    assert output.value == pytest.approx(value(b.value))
    coef = der(b.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_arccot_no_change(b):
    b_d = b.derivatives
    output = ad.arccot(b)
    assert b_d == b.derivatives
    
def test_arccot_sm(m):
    def value(x):
        return math.pi/2 - math.atan(x)
    output = ad.arccot(m)
    for i in range(len(m)):
        assert output.value[i] == pytest.approx(value(m[i]))
    assert output.derivatives == {}
        
def test_arccot_dm(dm):
    def value(x):
        return math.pi/2 - math.atan(x)
    def der(x):
        return -1/(1+x**2)
   
    output = ad.arccot(dm)
    for i in range(len(dm.value)):
        assert output.value[i] == pytest.approx(value(dm.value[i]))
        coef = der(dm.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dm.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dm.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dm.derivatives['z'][i])

    

###
# arcsec
###
def test_arcsec_s(sinv):
    def value(x):
        return math.acos(1/x)
    output = ad.arcsec(sinv)
    assert output.value == pytest.approx(value(sinv))
    assert output.derivatives == {}

def test_arcsec_d(binv):
    def value(x):
        return math.acos(1/x)
    def der(x):
        return 1/(abs(x)*math.sqrt(x**2-1))
    
    output = ad.arcsec(binv)
    assert output.value == pytest.approx(value(binv.value))
    coef = der(binv.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_arcsec_no_change(binv):
    b_d = binv.derivatives
    output = ad.arcsec(binv)
    assert b_d == binv.derivatives
    
def test_arcsec_sm(minv):
    def value(x):
        return math.acos(1/x)
    output = ad.arcsec(minv)
    for i in range(len(minv)):
        assert output.value[i] == pytest.approx(value(minv[i]))
    assert output.derivatives == {}
        
def test_arcsec_dm(dminv):
    def value(x):
        return math.acos(1/x)
    def der(x):
        return 1/(abs(x)*math.sqrt(x**2-1))
   
    output = ad.arcsec(dminv)
    for i in range(len(dminv.value)):
        assert output.value[i] == pytest.approx(value(dminv.value[i]))
        coef = der(dminv.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dminv.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dminv.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dminv.derivatives['z'][i])

    

###
# arccsc
###
def test_arccsc_s(sinv):
    def value(x):
        return math.asin(1/x)
    output = ad.arccsc(sinv)
    assert output.value == pytest.approx(value(sinv))
    assert output.derivatives == {}

def test_arccsc_d(binv):
    def value(x):
        return math.asin(1/x)
    def der(x):
        return -1/(abs(x)*math.sqrt(x**2-1))
    
    output = ad.arccsc(binv)
    assert output.value == pytest.approx(value(binv.value))
    coef = der(binv.value)
    assert output.derivatives == pytest.approx({'x':coef*1.2, 'y':coef*9.5, 'z':coef*5})
    
def test_arccsc_no_change(binv):
    b_d = binv.derivatives
    output = ad.arccsc(binv)
    assert b_d == binv.derivatives
    
def test_arccsc_sm(minv):
    def value(x):
        return math.asin(1/x)
    output = ad.arccsc(minv)
    for i in range(len(minv)):
        assert output.value[i] == pytest.approx(value(minv[i]))
    assert output.derivatives == {}
        
def test_arccsc_dm(dminv):
    def value(x):
        return math.asin(1/x)
    def der(x):
        return -1/(abs(x)*math.sqrt(x**2-1))
   
    output = ad.arccsc(dminv)
    for i in range(len(dminv.value)):
        assert output.value[i] == pytest.approx(value(dminv.value[i]))
        coef = der(dminv.value[i])
        assert output.derivatives['x'][i] == pytest.approx(coef*dminv.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(coef*dminv.derivatives['y'][i])
        assert output.derivatives['z'][i] == pytest.approx(coef*dminv.derivatives['z'][i])

    
