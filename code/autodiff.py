from collections import defaultdict
import numbers
import math

# powers/roots/exponential
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
            other = cls(None,0,defaultdict(float))
        return other
    
    def __add__(self, other):
        other = promote(other)
        output=self.emptyDual()
        
        output.value = self.value + other.value
        for k1 in self.derivatives:
            output.derivatives[k1] += self.derivatives[k1]
        for k2 in other.derivatives:
            output.derivatives[k2] += other.derivatives[k2]
        
        return output
    
    __radd__ = __add__
    
    def __sub__(self, other):
        try:
            y = autoDiff(self.val - other.val)
            y.deriv = self.deriv - other.deriv
        except AttributeError:
            y = autoDiff(self.val - other)
            y.deriv = self.deriv
        return y
            
    def __rsub__(self, other):
        try:
            y = autoDiff(self.val - other.val)
            y.deriv = self.deriv - other.deriv
        except AttributeError:
            y = autoDiff(other - self.val)
            y.deriv = -self.deriv
        return y
    def __mul__(self, other):
        other = promote(other)
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
        try:
            y = autoDiff(self.val/other.val)
            y.deriv = (self.deriv*other.val - self.val*other.deriv)/((other.val)**2)
        except AttributeError:
            y = autoDiff(self.val/other)
            y.deriv = self.deriv/other
        return y
    
    def __rtruediv__(self, other):
        try:
            y = autoDiff(self.val/other.val)
            y.deriv = (self.deriv*other.val - self.val*other.deriv)/((other.val)**2)
        except AttributeError:
            y = autoDiff(other/self.val)
            y.deriv = -other/((self.val)**2)*self.deriv
        return y            
    
    def __neg__(self):
        try:
            y = autoDiff(-self.val)
            y.deriv = -self.deriv
        except AttributeError:
            y = autoDiff(-self)
            y.deriv = -self
        return y
    
    def __pow__(self, other): #self^other => self = u(x) and other = v(x)
        try:
            y = autoDiff(self.val**(other.val))
            y.deriv = other.val*((self.val)**(other.val-1))*self.deriv + ((self.val)**(other.val))*(np.log(self.val))*other.deriv
        except AttributeError: #x^a:
            y = autoDiff(self.val**other)
            y.deriv = other*((self.val)**(other-1))*self.deriv
        return y
    
    def exp(other, self):
        try: # we may not necesarily implement this try part
            y = autoDiff(other.val**(self.val))
            y.deriv = other.val*((other.val)**(self.val-1))*other.deriv + ((other.val)**(self.val))*(np.log(other.val))*self.deriv
        except AttributeError:
            y = autoDiff(other**(self.val))
            y.deriv = other**(self.val)*np.log(other)*self.deriv
        return y
    
    def log(other, self):
        try:
            y = autoDiff(np.log(self.val)/np.log(other.val))
            y.deriv = (self.deriv*np.log(other.val)/self.val - other.deriv*np.log(self.val)/other.val)/(np.log(other.val)**2)
        except AttributeError:
            y = autoDiff(np.log(self.val)/np.log(other))
            y.deriv = 1/self.val/np.log(other)*self.deriv                
        return y
        
    def logx(self, other):#when base is a function of x
        try:
            y = autoDiff(np.log(other.val)/np.log(self.val))
            y.deriv = (other.deriv*np.log(self.val)/other.val - self.deriv*np.log(other.val)/self.val)/(np.log(self.val)**2)
        except AttributeError:
            y = autoDiff(np.log(other)/np.log(self.val))
            y.deriv = -np.log(other)*self.deriv/self.val/((np.log(self.val))**2)
        return y
        
    def sin(self):
        try:
            y = autoDiff(np.sin(self.val))
            y.deriv = np.cos(self.val)*self.deriv
        except AttributeError:
            y = autoDiff(np.sin(self))
            y.deriv = 0
        return y
        
    def cos(self):
        try:
            y = autoDiff(np.cos(self.val))
            y.deriv = -np.sin(self.val)*self.deriv
        except AttributeError:
            y = autoDiff(np.cos(self))
            y.deriv = 0
        return y
        
    def tan(self): #need to check 0 in denominator
        a = autoDiff.sin(self)
        b = autoDiff.cos(self)
        y = a/b
        return y
    
    def cot(self):
        a = autoDiff.tan(self)
        y = 1/a
        return y
    
    def sec(self):
        a = autoDiff.sin(self)
        y = 1/a
        return y
    
    def csc(self):
        a = autoDiff.cos(self)
        y = 1/a
        return y
    
    def arcsin(self): # must make sure self is strictly between -1 and 1, exclusive
        if type(self)==autoDiff and (self.val<-1 or self.val>1): # out of domain
            raise Exception('The value entering into arcsin function must be strictly within -1 and 1.')
            
        if type(self)!=autoDiff and (self<-1 or self>1):
            raise Exception('The value entering into arcsin function must be strictly within -1 and 1.')
            
        try:
            y = autoDiff(np.arcsin(self.val))
            y.deriv = 1/((1-self.val**2)**(0.5))*self.deriv
        except AttributeError:
            y = autoDiff(np.arcsin(self))
            y.deriv = 0
        return y
            
    def arccos(self):
        a = autoDiff.arcsin(self)
        y = np.pi/2 - a
        return y
    
    def arctan(self):
        a = self/((1+self**2)**0.5)
        y = autoDiff.arcsin(a)
        return y
    
    def arccot(self):
        a = autoDiff.arctan(self)
        y = np.pi/2 - a
        return y
    
    def arcsec(self):
        a = 1/self
        y = autoDiff.arccos(a)
        return y
    
    def arccsc(self):
        a = autoDiff.arcsec(self)
        y = np.pi/2 - a
        return y
    
    #### Do we need to implement hyperbolic function e.g. sinh, cosh, tanh, coth, etc.? ####
        

