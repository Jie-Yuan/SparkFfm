import collections
import itertools
import operator
import numpy
import copy
from pyspark import AccumulatorParam

from extensions import Extensions
import MixedData

DIMENSIONALITY = 5
ZERO_WEIGHT = None

class Weights:
	def __init__(self, integerColumns, categoricalColumns, uniqueCategoricals, dimensionality):
		self.integers = numpy.zeros((integerColumns, dimensionality))
		self.categoricals = numpy.zeros((categoricalColumns, uniqueCategoricals, dimensionality))

class Model:
	def __init__(self, data):
		self.weights = Weights(len(data.integerColumns), len(data.categoricalColumns), data.uniqueCategoricals, DIMENSIONALITY)
		
	@staticmethod
	def calculateLogLoss(rdd):
		#TODO
		a = 1

	###BROKEN
	def predict(self, data):
		
		weights = self.weights
		bcWeights = data.sc.broadcast(weights)
		
		def predictRow(row):
			
			del(row[0])
			row = [item if item is not None else 0 for item in row]
			
			tempWeights = list(bcWeights.value)
			
			#select weights for categorical features
			for i in range(len(row))[1:]:
				#check shape of weight to determine if categorical
				if len(tempWeights[i].shape) == 2:
					#this is a categorical feature, so index into weight array with value
					tempWeights[i] = tempWeights[i][row[i]]
					row[i] = 1

			#extract combinations
			lis = list(itertools.combinations(zip(tempWeights, row), 2))
			combinations = [zip(*tuple) for tuple in lis]
			
			return sum([numpy.dot(*combination[0]) * operator.mul(*combination[1]) for combination in combinations])
			
		return data.rdd.map(predictRow)

	def stochasticGradientDescent(self, data):
		integerShape = self.weights.integers.shape
		categoricalShape = self.weights.categoricals.shape
		
		accumIntegerWeights = data.sc.accumulator(numpy.zeros(integerShape), Extensions.ArrayAccumulatorParam())
		accumCategoricalWeights = data.sc.accumulator(numpy.zeros(categoricalShape), Extensions.ArrayAccumulatorParam())

		def accumulateWeight(row):
			integerDelta = numpy.ones(integerShape)
			categoricalDelta = numpy.ones(categoricalShape)
			
			#for array in [zeroIntegers, zeroCategoricals]:
			#	for element in numpy.nditer(array, op_flags=['readwrite']):
			#		element[...] = 1
			
			accumIntegerWeights.add(integerDelta)
			accumCategoricalWeights.add(categoricalDelta)
		
		data.rdd.foreach(accumulateWeight)
		
		self.weights.integer = accumIntegerWeights.value
		self.weights.integer = accumCategoricalWeights.value
		
		return 
	
	
class WeightAccumulatorParam(AccumulatorParam):
	def zero(self, initialValue):
		#unpack the dimensions of the initialValue and use to create new zero weight object
		(integerColumns, dimensionality) = initialValue.integers.shape
		(categoricalColumns, uniqueCategoricals, dimensionality) = initialValue.categoricals.shape
		return Weights(integerColumns, categoricalColumns, uniqueCategoricals, dimensionality)
	
	def addInPlace(self, a, b):
		a.integers = numpy.add(a.integers, b.integers)
		a.categoricals = numpy.add(a.categoricals, b.categoricals)
		
		return a