import pytest
import autodiff as ad

@pytest.fixture
def b():
    from autodiff import DualNumber
    b = DualNumber(None,-2/3,{'x':1.2, 'y':9.5, 'z':5})
    return b

@pytest.fixture
def s():
    return .42

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
	output = ad.sin(s)
	assert output.value == math.sin(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_sin_d(b):
	def der(x):
		return math.cos(x)
	
	output = ad.sin(b)
	assert output.value == math.sin(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_sin_no_change(b):
    b_d = b.derivatives
    output = ad.sin(b)
    assert b_d == b.derivatives
	

###
# cos
###
def test_cos_s(s):
	output = ad.cos(s)
	assert output.value == math.cos(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_cos_d(b):
	def der(x):
		return -math.sin(x)
	
	output = ad.cos(b)
	assert output.value == math.cos(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_cos_no_change(b):
    b_d = b.derivatives
    output = ad.cos(b)
    assert b_d == b.derivatives
	

###
# tan
###
def test_tan_s(s):
	output = ad.tan(s)
	assert output.value == math.tan(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_tan_d(b):
	def der(x):
		return math.sec(x)**2
	
	output = ad.tan(b)
	assert output.value == math.tan(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_tan_no_change(b):
    b_d = b.derivatives
    output = ad.tan(b)
    assert b_d == b.derivatives
	

###
# cot
###
def test_cot_s(s):
	output = ad.cot(s)
	assert output.value == math.cot(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_cot_d(b):
	def der(x):
		return -math.csc(x)**2
	
	output = ad.cot(b)
	assert output.value == math.cot(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_cot_no_change(b):
    b_d = b.derivatives
    output = ad.cot(b)
    assert b_d == b.derivatives
	

###
# sec
###
def test_sec_s(s):
	output = ad.sec(s)
	assert output.value == math.sec(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_sec_d(b):
	def der(x):
		return math.sec(x)*math.tan(x)
	
	output = ad.sec(b)
	assert output.value == math.sec(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_sec_no_change(b):
    b_d = b.derivatives
    output = ad.sec(b)
    assert b_d == b.derivatives
	

###
# csc
###
def test_csc_s(s):
	output = ad.csc(s)
	assert output.value == math.csc(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_csc_d(b):
	def der(x):
		return -math.csc(x)*math.cot(x)
	
	output = ad.csc(b)
	assert output.value == math.csc(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_csc_no_change(b):
    b_d = b.derivatives
    output = ad.csc(b)
    assert b_d == b.derivatives
	

###
# arcsin
###
def test_arcsin_s(s):
	output = ad.arcsin(s)
	assert output.value == math.arcsin(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_arcsin_d(b):
	def der(x):
		return 1/math.sqrt(1-x**2)
	
	output = ad.arcsin(b)
	assert output.value == math.arcsin(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_arcsin_no_change(b):
    b_d = b.derivatives
    output = ad.arcsin(b)
    assert b_d == b.derivatives
	

###
# arccos
###
def test_arccos_s(s):
	output = ad.arccos(s)
	assert output.value == math.arccos(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_arccos_d(b):
	def der(x):
		return -1/math.sqrt(1-x**2)
	
	output = ad.arccos(b)
	assert output.value == math.arccos(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_arccos_no_change(b):
    b_d = b.derivatives
    output = ad.arccos(b)
    assert b_d == b.derivatives
	

###
# arctan
###
def test_arctan_s(s):
	output = ad.arctan(s)
	assert output.value == math.arctan(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_arctan_d(b):
	def der(x):
		return 1/(1+x**2)
	
	output = ad.arctan(b)
	assert output.value == math.arctan(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_arctan_no_change(b):
    b_d = b.derivatives
    output = ad.arctan(b)
    assert b_d == b.derivatives
	

###
# arccot
###
def test_arccot_s(s):
	output = ad.arccot(s)
	assert output.value == math.arccot(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_arccot_d(b):
	def der(x):
		return -1/(1+x**2)
	
	output = ad.arccot(b)
	assert output.value == math.arccot(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_arccot_no_change(b):
    b_d = b.derivatives
    output = ad.arccot(b)
    assert b_d == b.derivatives
	

###
# arcsec
###
def test_arcsec_s(s):
	output = ad.arcsec(s)
	assert output.value == math.arcsec(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_arcsec_d(b):
	def der(x):
		return 1/(math.abs(x)*math.sqrt(x**2-1))
	
	output = ad.arcsec(b)
	assert output.value == math.arcsec(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_arcsec_no_change(b):
    b_d = b.derivatives
    output = ad.arcsec(b)
    assert b_d == b.derivatives
	

###
# arccsc
###
def test_arccsc_s(s):
	output = ad.arccsc(s)
	assert output.value == math.arccsc(s)
	assert output.derivatives == {'x':0,'y':0,'z':0}

def test_arccsc_d(b):
	def der(x):
		return -1/(math.abs(x)*math.sqrt(x**2-1))
	
	output = ad.arccsc(b)
	assert output.value == math.arccsc(b.value)
	assert output.derivatives == {'x':der(1.2), 'y':der(9.5), 'z':der(5)}
	
def test_arccsc_no_change(b):
    b_d = b.derivatives
    output = ad.arccsc(b)
    assert b_d == b.derivatives
	
