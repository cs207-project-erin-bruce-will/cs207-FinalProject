from collections import defaultdict
import numbers
import math

# TODO: powers/roots/exponential
class DualNumber():
    def __init__(self, name, value, derivatives=None):
        # ideally, we should block users from using the derivtives interface. May require separate classes
        self.value = value
        if name is not None:
            self.derivatives = defaultdict(float)
            self.derivatives[name] = 1
        else:
            self.derivatives = derivatives
            
    @classmethod
    def emptyDual(cls):
        return cls(None,0,defaultdict(float))
        
    @classmethod
    def promote(cls, other):
        if isinstance(other,numbers.Number):
            other = cls(None,other,defaultdict(float))
        return other
    
    def __add__(self, other):
        other = self.promote(other)
        output=self.emptyDual()
        
        output.value = self.value + other.value
        for k1 in self.derivatives:
            output.derivatives[k1] += self.derivatives[k1]
        for k2 in other.derivatives:
            output.derivatives[k2] += other.derivatives[k2]
        
        return output
    
    __radd__ = __add__
    
    def __sub__(self, other):
        other = self.promote(other)
        output = self.emptyDual()
        
        output.value = self.value - other.value
        for k1 in self.derivatives:
            output.derivatives[k1] += -self.derivatives[k1]
        for k2 in other.derivatives:
            output.derivatives[k2] += -other.derivatives[k2]

        return output
    
    __rsub__ = __sub__
    
    def __mul__(self, other):
        other = self.promote(other)
        output=self.emptyDual()
        
        output.value = self.value*other.value
        
        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += self.value*other.derivatives[k2]
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += other.value*self.derivatives[k1]
            
        return output
        
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        other = self.promote(other)
        output = self.emptyDual()
        
        output.value = self.value/other.value

        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += -self.value*other.derivatives[k2]/((other.value)**2)
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += other.value*self.derivatives[k1]/((other.value)**2)
            
        return output
    
    __rtruediv__ = __truediv__
    
    def __neg__(self):
        output = self.emptyDual()
        
        output.value = -self.value
        output.derivatives = -self.derivatives

        return output

    

    def __pow__(self, other): #self^other => self = u(x) and other = v(x)
        other = self.promote(other)
        output = self.emptyDual()
        
        output.value = self.value**other.value

        # real part of first parent distributes
        for k2 in other.derivatives:
            output.derivatives[k2] += (self.value**other.value)*math.log(self.value)*other.derivatives[k2]
        
        # real part of the second parent distributes
        for k1 in self.derivatives:
            output.derivatives[k1] += other.value*(self.value**(other.value-1))*self.derivatives[k1]
            
        return output
 

####
# End of DualNumber class; start of module's functions
#### 

#### I think we probably do not need this exp method because __power__ has achieve the goal by design ####
def exp(other, self):
    pass

    
def ln(self): #natural log
    output = self.emptyDual()
        
    output.value = math.log(self.value)
        
    # real part of the second parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += self.derivatives[k1]/self.value
            
    return output        
    
def log(self, other): #self is base
    other = self.promote(other)
    output = self.emptyDual()
        
    output = ln(other)/ln(self)
            
    return output

    
    #### We probably do not need this logx method, either ####
def logx(self, other):#when base is a function of x
    pass
    

def sin(self):
    output = self.emptyDual()
    output.value = math.sin(self.value)
        
    # real part of the first parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += math.cos(self.value)*self.derivatives[k1]
            
    return output


def cos(self):
    output = self.emptyDual()
    output.value = math.cos(self.value)
        
    # real part of the first parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += -math.sin(self.value)*self.derivatives[k1]
            
    return output


def tan(self): #need to check 0 in denominator
    output = self.emptyDual()
    output = sin(self)/cos(self)
            
    return output


def cot(self):
    output = self.emptyDual()
    output = cos(self)/sin(self)
            
    return output


def sec(self):
    output = self.emptyDual()
    output = 1/sin(self)
            
    return output


def csc(self):
    output = self.emptyDual()
    output = 1/cos(self)
            
    return output


def arcsin(self): # must make sure self is strictly between -1 and 1, exclusive
    output = self.emptyDual()
    if self.value<-1 or self.value>1: # out of domain
        raise Exception('The value entering into arcsin function must be strictly within -1 and 1.')

    output.value = math.asin(self.value)
        
    # real part of the first parent distributes
    for k1 in self.derivatives:
        output.derivatives[k1] += 1/math.sqrt(1-self.value**2)*self.derivatives[k1]
            
    return output


def arccos(self):
    output = self.emptyDual()
    output = math.pi/2 - arcsin(self)
        
    return output


def arctan(self):
    output = self.emptyDual()
    output = arcsin(self/math.sqrt(1+self**2))
        
    return output


def arccot(self):
    output = self.emptyDual()
    output = math.pi/2 - arctan(self)
        
    return output


def arcsec(self):
    output = self.emptyDual()
    output = arccos(1/self)
        
    return output


def arccsc(self):
    output = self.emptyDual()
    output = math.pi/2 - arcsec(self)
        
    return output
