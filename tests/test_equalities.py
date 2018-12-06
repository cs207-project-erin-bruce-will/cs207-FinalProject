import pytest
import autodiff as ad
import math
import warnings

#TODO: Test the warnings?

@pytest.fixture
def a():
    from autodiff import DualNumber
    a = DualNumber._from_dict(3.5,{'y':4,'x':3})
    return a

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
# Equality ==
###
def test__eq__false():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) == 3.1
    assert result == False
    
def test__eq__true():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) == 3.5
    assert result == True
    
def test__eq__dual():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) == ad.DualNumber._from_dict(3.5,{'x':3})
    assert result == True
    
def test__eq__dual_false():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) == ad.DualNumber._from_dict(1.5,{'y':4,'x':3})
    assert result == False
    
###
# Not Equal !=
###
def test__ne__true():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) != 3.1
    assert result == True
    
def test__ne__false():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) !=3.5
    assert result == False
    
def test__ne__dual_false():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) != ad.DualNumber._from_dict(3.5,{'x':3})
    assert result == False   
    
def test__ne__dual_true():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) != ad.DualNumber._from_dict(1.5,{'y':4,'x':3})
    assert result == True
    
###
# Less Than or Equal To
###
def test__le__false():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) <= 3.1
    assert result == False
    
def test__le__true(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) <=3.6
    assert result == True
    
def test__le__dual(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) <= ad.DualNumber._from_dict(3.5,{'x':3})
    assert result == True   
    
def test__le__dual_false(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) <= ad.DualNumber._from_dict(1.5,{'y':4,'x':3})
    assert result == False
    
###
# Less Than
###
def test__lt__false():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) < 3.1
    assert result == False
    
def test__lt__true(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) < 3.6
    assert result == True
    
def test__lt__dual(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) < ad.DualNumber._from_dict(3.5,{'x':3})
    assert result == False   
    
def test__lt__dual_false(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) <= ad.DualNumber._from_dict(1.5,{'y':4,'x':3})
    assert result == False
    
###
# Greater Than or Equal To
###
def test__ge__true():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) >= 3.1
    assert result == True
    
def test__ge__false(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) >=3.6
    assert result == False
    
def test__ge__dual(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) >= ad.DualNumber._from_dict(3.5,{'x':3})
    assert result == True   
    
def test__ge__dual_true(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) >= ad.DualNumber._from_dict(1.5,{'y':4,'x':3})
    assert result == True
    
 ###
# Greater Than 
###   
    
def test__gt__true():
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) > 3.1
    assert result == True
    
def test__gt__false(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) > 3.6
    assert result == False
    
def test__gt__dual(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) > ad.DualNumber._from_dict(3.5,{'x':3})
    assert result == False
    
def test__gt__dual_true(a):
    result = ad.DualNumber._from_dict(3.5,{'y':4,'x':3}) > ad.DualNumber._from_dict(1.5,{'y':4,'x':3})
    assert result == True
    
