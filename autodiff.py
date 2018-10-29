from collections import defaultdict
import numbers

class DualNumber:

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
	
	def __mul__(self, other):
		output=self.emptyDual()
		
		output.value = self.value*other.value
		
		# real part of first parent distributes
		for k2 in other.derivatives:
			output.derivatives[k2] += self.value*other.derivatives[k2]
		
		# real part of the second parent distributes
		for k1 in self.derivatives:
			output.derivatives[k1] += other.value*self.derivatives[k1]
			
		return output
	
	def __rmul__(self,other):
		return self.__mul__(other)
	
	def __add__(self, other):
		output=self.emptyDual()
		
		output.value = self.value + other.value
		for k1 in self.derivatives:
			output.derivatives[k1] += self.derivatives[k1]
		for k2 in other.derivatives:
			output.derivatives[k2] += other.derivatives[k2]
		
		return output