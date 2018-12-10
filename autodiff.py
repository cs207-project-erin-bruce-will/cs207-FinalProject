from collections import defaultdict
import numbers
import numpy as np
import warnings


class DualNumber():
    """This class contains the functions that return dual numbers."""
    
    def __init__(self, name, value):
        # ideally, we should block users from using the derivtives interface. May require separate classes
        """Declare a dual number variable
        
        INPUTS:
        =======
        name: a string declaring the name of the variable, ie 'x'
        value: an integer or float declaring the current value of the variable
        
        RETURNS:
        =======
        self.DualNumber: object
            A DualNuber object
            
        EXAMPLES:
        =======
       
        >>> DualNumber('x',4).value
        4
        
        >>> DualNumber('x',4).derivatives
        defaultdict(<class 'float'>, {'x': 1})
        """
        
        if isinstance(value,numbers.Number):
            self.value = value
            self.derivatives = defaultdict(float)
            if not isinstance(name,str):
                raise TypeError("name for input must be a string (when value is a single number)")
            self.derivatives[name]=1
        
        elif isinstance(value, np.ndarray):
            self.value = value
            self.derivatives = defaultdict(self.np_closure(value.shape))
            position_list = np.unravel_index(range(value.size),value.shape)
            for cur_index in range(value.size):
                
                indices = [dim[cur_index] for dim in position_list]
                
                der = np.zeros(value.shape)
                der[tuple(indices)] = 1
                
                extended_name = name+repr(indices)
                
                self.derivatives[extended_name] = der
        else:
            raise TypeError("Couldn't convert input of type {}".format(type(value)))
    
               
    @staticmethod
    def np_closure(shape):
        """
        @staticmethod used to create a function that returns a np array of a given shape.
        """
        def inner_func():
            return np.zeros(shape)
        return inner_func
        
    @classmethod
    def promote(cls, other):
        """
        @classmethod to promote DualNumber object.
        """
        # if already a dual number, just return it
        if isinstance(other, DualNumber):
            return other
        
        output = DualNumber('',0) # build a skeleton we'll overwrite
        if isinstance(other,numbers.Number):
            output.value = other
            output.derivatives = defaultdict(float)
        elif isinstance(other, np.ndarray):
            output.value = other
            output.derivatives = defaultdict(cls.np_closure(other.shape))
        else:
            raise TypeError("Couldn't promote input of type {}".format(type(other)))
        return output
    
    @classmethod
    def _from_dict(cls, val, deriv):
        """
        @classmethod explicitly create the components of a dual number.
        """
        output = cls.promote(val)
        for k,v in deriv.items():
            output.derivatives[k]=v
        return output
    
    def __add__(self, other):
        """Adds DualNumber object to a value or another DualNumber, and returns DualNumber object with updated values and derivatives."""
        other = self.promote(other)
        
        output = self.promote(self.value + other.value)
                
        for k1 in self.derivatives:
            output.derivatives[k1] += self.derivatives[k1]
        for k2 in other.derivatives:
            output.derivatives[k2] += other.derivatives[k2]
        
        return output
    
    __radd__ = __add__
    
    def __sub__(self, other):
        """Subtracts DualNumber object from a value or another DualNumber, and returns a new DualNumber object."""
        other = self.promote(other)
        
        output = self.promote(self.value - other.value)
        
        for k1 in self.derivatives:
            output.derivatives[k1] += self.derivatives[k1]
        for k2 in other.derivatives:
            output.derivatives[k2] += -other.derivatives[k2]

        return output
    
    def __rsub__(self, other):
        """Subtracts DualNumber object from a value or another DualNumber, and returns a new DualNumber object."""
        other = self.promote(other)
        
        output = self.promote(other.value - self.value)
        
        for k1 in self.derivatives:
            output.derivatives[k1] += -self.derivatives[k1]
        for k2 in other.derivatives:
            output.derivatives[k2] += other.derivatives[k2]

        return output         
    
    def __mul__(self, other):
        """Multiplies DualNumber object with a value or another DualNumber, and returns a new DualNumber object."""
        other = self.promote(other)
        
        output = self.promote(self.value*other.value)
        
        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += self.value*other.derivatives[k2]
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += other.value*self.derivatives[k1]
            
        return output
        
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        """Divides DualNumber object by a value or another DualNumber, and returns a new DualNumber object."""
        other = self.promote(other)
        
        output = self.promote(self.value/other.value)

        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += -self.value*other.derivatives[k2]/((other.value)**2)
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += other.value*self.derivatives[k1]/((other.value)**2)
            
        return output
    
    def __rtruediv__(self, other):
        """Divides DualNumber object by a value or another DualNumber, and returns a new DualNumber object."""
        other = self.promote(other)
        
        output = self.promote(other.value/self.value)

        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += self.value*other.derivatives[k2]/((self.value)**2)
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += -other.value*self.derivatives[k1]/((self.value)**2)
            
        return output         
    
    def __neg__(self):
        """Makes DualNumber object negative, and returns a new DualNumber object reflecting the change."""
        output = self.promote(-self.value)
        
        for k1 in self.derivatives:
            output.derivatives[k1] += -self.derivatives[k1]

        return output

    

    def __pow__(self, other): #self^other => self = u(x) and other = v(x)
        """Raises DualNumber object to specificed power, and returns a new DualNumber object reflecting the change."""
        other = self.promote(other)
        
        output = self.promote(self.value**other.value)

        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += (self.value**other.value)*np.log(self.value)*other.derivatives[k2]
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += other.value*(self.value**(other.value-1))*self.derivatives[k1]
            
        return output
    
    def __rpow__(self, other):
        """Raises DualNumber object to specificed power, and returns a new DualNumber object reflecting the change."""
        other = self.promote(other)
        
        output = self.promote(other.value**self.value)

        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += self.value*(other.value**(self.value-1))*other.derivatives[k2]
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += (other.value**self.value)*np.log(other.value)*self.derivatives[k1]

        return output
    
    def __eq__(self, other):
        """Checks if DualNumber object is equal to a specific value.
        
        Note: This functions only checks the .value part of the dual number.
        It does note check if the derivatives are equal.
        """
        if isinstance(other, DualNumber):
            if self.value == other.value:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value == other.value
        else:
            if self.value == other:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value == other
        
    def __le__(self, other):
        """Checks if DualNumber object is less than or equal to a specific value.
        
        Note: This functions only checks the .value part of the dual number.
        It does note check if the derivatives are equal.
        """
        if isinstance(other, DualNumber):
            if self.value == other.value:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value <= other.value
        else:
            if self.value == other:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value <= other
        
    def __lt__(self, other):
        """Checks if DualNumber object is less than a specific value.
        
        Note: This functions only checks the .value part of the dual number.
        It does note check if the derivatives are equal.
        """
        if isinstance(other, DualNumber):
            if self.value == other.value:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value < other.value
        else:
            if self.value == other:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value < other
    
    def __ge__(self, other):
        """Checks if DualNumber object is greater than or equal to a specific value.
        
        Note: This functions only checks the .value part of the dual number.
        It does note check if the derivatives are equal.
        """
        if isinstance(other, DualNumber):
            if self.value == other.value:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value >= other.value
        else:
            if self.value == other:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value >= other
            
    def __gt__(self, other):
        """Checks if DualNumber object is greater than a specific value.
        
        Note: This functions only checks the .value part of the dual number.
        It does note check if the derivatives are equal.
        """
        if isinstance(other, DualNumber):
            if self.value == other.value:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value > other.value
        else:
            if self.value == other:
                warnings.warn("Computation is close to a branch. Derivatives may not be accurate.")
            return self.value > other
        

####
# End of DualNumber class; start of module's functions
#### 

def sqrt(self): #square root function overload
    """
    Takes the square root of a DualNumber object and returns DualNumber object with updated value and derivatives.
    """

    output = self**0.5

    return output
    
def exp(self): #natural exponential
    """
    Raises DualNumber object to a specified value and returns a DualNumber object with updated value and derivatives.
    """
    
    base = np.exp(1)
    output = base**self

    return output
    
def ln(self): #natural log
    """
    Takes the natural log of DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    output= self.promote(np.log(self.value))
        
    # real part of the second parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += self.derivatives[k1]/self.value
            
    return output         
    
def log(x, base): #log_other(self) and other is base
    """
    Takes the log of DualNumber object at a specified value and returns a DualNumber object with updated value and derivatives.
    """
    x = DualNumber.promote(x)
    base =  DualNumber.promote(base)

    if np.any(x.value<=0): # out of domain
        raise Exception('The domain of logarithm function is any positive number.')
        
    if np.any(base.value<=0): # base must be something positive
        raise Exception('The base of logarithm function is any positive number.')
        
    return ln(x)/ln(base)

def sin(self):
    """
    Takes the sine of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = self.promote(np.sin(self.value))
        
    # real part of the first parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += np.cos(self.value)*self.derivatives[k1]
            
    return output


def cos(self):
    """
    Takes the cosine of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = self.promote(np.cos(self.value))
        
    # real part of the first parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += -np.sin(self.value)*self.derivatives[k1]
            
    return output


def tan(self): #need to check 0 in denominator
    """
    Takes the tangent of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = sin(self)/cos(self)
            
    return output


def cot(self):
    """
    Takes the cotangent of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = cos(self)/sin(self)
            
    return output


def sec(self):
    """
    Takes the secant of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = 1/cos(self)
            
    return output


def csc(self):
    """
    Takes the cosecant of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = 1/sin(self)
            
    return output


def arcsin(self): # must make sure self is strictly between -1 and 1, exclusive
    """
    Takes the inverse sine of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    
    Note: .value will only fall between -1 and 1
    """
    self = DualNumber.promote(self)
    if np.any(self.value<-1) or np.any(self.value>1): # out of domain
        raise Exception('The value entering into arcsin function must be strictly within -1 and 1.')

    output = self.promote(np.arcsin(self.value))
        
    # real part of the first parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += 1/np.sqrt(1-self.value**2)*self.derivatives[k1]
            
    return output


def arccos(self):
    """
    Takes the inverse cosine of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = np.pi/2 - arcsin(self)
        
    return output


def arctan(self):
    """
    Takes the inverse tangent of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = arcsin(self/sqrt(1+self**2))
        
    return output


def arccot(self):
    """
    Takes the inverse cotangent of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = np.pi/2 - arctan(self)
        
    return output


def arcsec(self):
    """
    Takes the inverse secant of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = arccos(1/self)
        
    return output

def arccsc(self):
    """
    Takes the inverse cosecant of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = np.pi/2 - arcsec(self)
        
    return output

def sinh(self):
    """
    Takes the hyperbolic sine of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = exp(self)/2 - exp(-self)/2

    return output

def cosh(self):
    """
    Takes the hyperbolic cosine of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = exp(self)/2 + exp(-self)/2

    return output

def tanh(self):
    """
    Takes the hyperbolic tangent of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = sinh(self)/cosh(self)

    return output

def coth(self):
    """
    Takes the hyperbolic cotangent of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = cosh(self)/sinh(self)

    return output

def sech(self):
    """
    Takes the hyperbolic secant of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = 1/cosh(self)

    return output

def csch(self):
    """
    Takes the hyperbolic cosecant of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = 1/sinh(self)

    return output

def logistic(self):
    """
    Takes the logistic of a DualNumber object and returns a DualNumber object with updated value and derivatives.
    """
    self = DualNumber.promote(self)
    output = exp(self)/(exp(self)+1)

    return output
