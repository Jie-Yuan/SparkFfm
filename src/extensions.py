from pyspark import AccumulatorParam
from operator import add
import numpy

class Extensions:

	class ListAccumulatorParam(AccumulatorParam):
		def zero(self, initialValue):
			return [0] * len(initialValue)
		
		def addInPlace(self, a, b):
			#None-safe addition (None treated as zero)
			a = [0 if x == None else x for x in a]
			b = [0 if x == None else x for x in b]
			return map(add, a, b)
	
	class ArrayAccumulatorParam(AccumulatorParam):
		def zero(self, initialValue):
			return numpy.zeros(initialValue.shape)
		
		def addInPlace(self, a, b):
			return numpy.add(a, b)
	
	class ArgumentException(Exception):
		pass
	
