import pytest
import autodiff as ad
import numpy as np
import math

@pytest.fixture
def m_array():
    from autodiff import DualNumber
    m = DualNumber._from_dict(np.array([3,2,1]),{'y':np.array([4,-5,6]),'x':np.array([8,7,-9])})
    return m

@pytest.fixture
def n_array():
    from autodiff import DualNumber
    n = DualNumber._from_dict(np.array([-2,3,6]),{'y':np.array([4,-5,1]),'x':np.array([-1,3,-9])})
    return n

@pytest.fixture
def p_array():
    from autodiff import DualNumber
    n = DualNumber._from_dict(np.array([2,3,6]),{'y':np.array([4,-5,1]),'x':np.array([-1,3,-9])})
    return n

@pytest.fixture
def rect_array():
    from autodiff import DualNumber
    rect_array = DualNumber._from_dict(np.array([[3,2,1],[6,2,2]]),{'y':np.array([[4,-5,6],[0,0,2]]),'x':np.array([[8,7,-9],[3,2,5]])})
    return rect_array

@pytest.fixture
def b():
    from autodiff import DualNumber
    b = DualNumber._from_dict(2,{'x':1.2, 'y':9.5, 'z':5})
    return b

@pytest.fixture
def s():
    return 4.2
	
@pytest.fixture
def s2():
    return 3.1

###
# Creation
###
def test_creation_arr():
    x = ad.DualNumber('x',np.array([10,12]))
    assert np.all(x.value == np.array([10,12]))
    assert np.all(x.derivatives['x[0]'] == np.array([1,0]))
    assert np.all(x.derivatives['x[1]'] == np.array([0,1]))


###
# Addition
###
def test_add_ms(m_array,s):
    output = m_array+s
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]+s
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i]
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i]
    
def test_add_dm(m_array,b):
    output = b+m_array
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]+b.value
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i] + b.derivatives['y']
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i] + b.derivatives['x']

def test_add_mm(m_array,n_array):
    output = m_array+n_array
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]+n_array.value[i]
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i] + n_array.derivatives['y'][i]
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i] + n_array.derivatives['x'][i]

###
# Subtraction
###
def test_sub_ms(m_array,s):
    output = m_array-s
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]-s
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i]
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i]
    
def test_sub_dm(m_array,b):
    output = b-m_array
    for i in range(len(m_array.value)):
        assert output.value[i] == b.value-m_array.value[i]
        assert output.derivatives['y'][i] == b.derivatives['y'] - m_array.derivatives['y'][i]
        assert output.derivatives['x'][i] == b.derivatives['x'] - m_array.derivatives['x'][i]

def test_sub_mm(m_array,n_array):
    output = m_array-n_array
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]-n_array.value[i]
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i] - n_array.derivatives['y'][i]
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i] - n_array.derivatives['x'][i]
        
        
###
# Multiplication
###
def test_mul_ms(m_array,s):
    output = m_array*s
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]*s
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i]*s
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i]*s
    
def test_mul_dm(m_array,b):
    output = b*m_array
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]*b.value
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i]*b.value + b.derivatives['y']*m_array.value[i]
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i]*b.value + b.derivatives['x']*m_array.value[i]

def test_mul_mm(m_array,n_array):
    output = m_array*n_array
    for i in range(len(m_array.value)):
        assert output.value[i] == m_array.value[i]*n_array.value[i]
        assert output.derivatives['y'][i] == m_array.derivatives['y'][i]*n_array.value[i] + n_array.derivatives['y'][i]*m_array.value[i]
        assert output.derivatives['x'][i] == m_array.derivatives['x'][i]*n_array.value[i] + n_array.derivatives['x'][i]*m_array.value[i]
        
###
# Division
###
def test_div_ms(m_array,s):
    output = m_array/s
    for i in range(len(m_array.value)):
        assert output.value[i] == pytest.approx(m_array.value[i]/s)
        denom = s
        assert output.derivatives['y'][i] == pytest.approx((m_array.derivatives['y'][i]*s)/denom**2)
        assert output.derivatives['x'][i] == pytest.approx((m_array.derivatives['x'][i]*s)/denom**2)
        
def test_div_dm(m_array,b):
    output = b/m_array
    for i in range(len(m_array.value)):
        assert output.value[i] == b.value/m_array.value[i]
        denom = m_array.value[i]
        assert output.derivatives['y'][i] == pytest.approx((b.derivatives['y']*m_array.value[i] - m_array.derivatives['y'][i]*b.value)/denom**2)
        assert output.derivatives['x'][i] == pytest.approx((b.derivatives['x']*m_array.value[i] - m_array.derivatives['x'][i]*b.value)/denom**2)
        
def test_div_mm(m_array,n_array):
    output = m_array/n_array
    for i in range(len(m_array.value)):
        assert output.value[i] == pytest.approx(m_array.value[i]/n_array.value[i])
        denom = n_array.value[i]
        assert output.derivatives['y'][i] == pytest.approx((m_array.derivatives['y'][i]*n_array.value[i] - n_array.derivatives['y'][i]*m_array.value[i])/denom**2)
        assert output.derivatives['x'][i] == pytest.approx((m_array.derivatives['x'][i]*n_array.value[i] - n_array.derivatives['x'][i]*m_array.value[i])/denom**2)
        
###
# Exponents
###
def test_exponents_ms(m_array,s):
    output = m_array**s
    for i in range(len(m_array.value)): 
        assert output.value[i] == m_array.value[i]**s
        m  = m_array.value[i]**s
        ma = s/m_array.value[i]
        mb = math.log(m_array.value[i])
        assert output.derivatives['x'][i] == m*ma*m_array.derivatives['x'][i]
        assert output.derivatives['y'][i] == m*ma*m_array.derivatives['y'][i]

def test_exponents_md(m_array,b):
    output = m_array**b
    for i in range(len(m_array.value)): 
        assert output.value[i] == m_array.value[i]**b.value
        m  = m_array.value[i]**b.value
        ma = b.value/m_array.value[i]
        mb = math.log(m_array.value[i])
        assert output.derivatives['x'][i] == m*ma*m_array.derivatives['x'][i]+m*mb*b.derivatives['x']
        assert output.derivatives['y'][i] == m*ma*m_array.derivatives['y'][i]+m*mb*b.derivatives['y']

def test_exponents_dm(b,p_array):
    output = b**p_array
    for i in range(len(p_array.value)): 
        assert output.value[i] == b.value**p_array.value[i]
        m  = b.value**p_array.value[i]
        ma = p_array.value[i]/b.value
        mb = math.log(b.value)
        assert output.derivatives['x'][i] == m*ma*b.derivatives['x']+m*mb*p_array.derivatives['x'][i]
        assert output.derivatives['y'][i] == m*ma*b.derivatives['y']+m*mb*p_array.derivatives['y'][i]
        
def test_exponents_mm(m_array,p_array):
    output = m_array**p_array
    for i in range(len(m_array.value)): 
        assert output.value[i] == m_array.value[i]**p_array.value[i]
        m  = m_array.value[i]**p_array.value[i]
        ma = p_array.value[i]/m_array.value[i]
        mb = math.log(m_array.value[i])
        assert output.derivatives['x'][i] == m*ma*m_array.derivatives['x'][i]+m*mb*p_array.derivatives['x'][i]
        assert output.derivatives['y'][i] == m*ma*m_array.derivatives['y'][i]+m*mb*p_array.derivatives['y'][i]

        
###
# logs
###
def test_log_ms(m_array,s):
    output = ad.log(m_array,s)
    for i in range(len(m_array.value)):  
        assert output.value[i] == math.log(m_array.value[i],s)
        m  = 1/math.log(s)**2
        ma = math.log(s)/m_array.value[i]
        assert output.derivatives['x'][i] == pytest.approx(m*ma*m_array.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(m*ma*m_array.derivatives['y'][i])


def test_log_md(m_array,b):
    output = ad.log(m_array,b)
    for i in range(len(m_array.value)):  
        assert output.value[i] == math.log(m_array.value[i],b.value)
        m  = 1/math.log(b.value)**2
        ma = math.log(b.value)/m_array.value[i]
        mb = -math.log(m_array.value[i])/b.value
        assert output.derivatives['x'][i] == pytest.approx(m*ma*m_array.derivatives['x'][i]+m*mb*b.derivatives['x'])
        assert output.derivatives['y'][i] == pytest.approx(m*ma*m_array.derivatives['y'][i]+m*mb*b.derivatives['y'])

def test_log_dm(b,p_array):
    output = ad.log(b,p_array)
    for i in range(len(p_array.value)):  
        assert output.value[i] == math.log(b.value,p_array.value[i])
        m  = 1/math.log(p_array.value[i])**2
        ma = math.log(p_array.value[i])/b.value
        mb = -math.log(b.value)/p_array.value[i]
        assert output.derivatives['x'][i] == pytest.approx(m*ma*b.derivatives['x']+m*mb*p_array.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(m*ma*b.derivatives['y']+m*mb*p_array.derivatives['y'][i])
        
def test_log_mm(m_array,p_array):
    output = ad.log(m_array,p_array)
    for i in range(len(m_array.value)):  
        assert output.value[i] == math.log(m_array.value[i],p_array.value[i])
        m  = 1/math.log(p_array.value[i])**2
        ma = math.log(p_array.value[i])/m_array.value[i]
        mb = -math.log(m_array.value[i])/p_array.value[i]
        assert output.derivatives['x'][i] == pytest.approx(m*ma*m_array.derivatives['x'][i]+m*mb*p_array.derivatives['x'][i])
        assert output.derivatives['y'][i] == pytest.approx(m*ma*m_array.derivatives['y'][i]+m*mb*p_array.derivatives['y'][i])
        
###
# Transpose
###
def test_T(rect_array):
    output = ad.T(rect_array)
    nrow, ncol = output.value.shape
    for i in range(nrow):
        for j in range(ncol):
            assert output.value[i,j] == rect_array.value[j,i]
            assert output.derivatives['y'][i,j] == rect_array.derivatives['y'][j,i]
            assert output.derivatives['x'][i,j] == rect_array.derivatives['x'][j,i]
              
        
        
###
# Dot product
###
