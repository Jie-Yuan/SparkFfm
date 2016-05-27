import math
import hashlib
from pyspark import *
from operator import add

from extensions import Extensions

MAX_UNIQUE_CATEGORICAL_FEATURES = long(10 ** 4)

class PreProcessing:
	@staticmethod
	def normaliseInts(data):

		#calculate sum of each int feature
		sum = data.accumulateOnColumns(lambda value: value, data.integerColumns)
		
		#calculate mean of each int feature
		count = data.rdd.count()
		means = [sum[col] / float(count - data.noneCount[col]) for col in data.allColumns]
		
		print("means: " + str(means))
		
		#subtract mean from each integer feature
		data.mapOnColumnsWithParam(lambda value, mean: value - mean if value != None else None, means, data.integerColumns)
		
		#calculate sum of squared difference from the mean
		squareSum = data.accumulateOnColumns(lambda value: value ** 2 if value != None else None, data.integerColumns)
		
		#calculate standard deviation of each int feature
		stdDevs = [math.sqrt(squareSum[col] / float(count - data.noneCount[col])) for col in data.allColumns]
		
		#divide each integer feature by SD
		data.mapOnColumnsWithParam(lambda value, sd: float(value) / sd if value != None and sd != 0 else value, stdDevs, data.integerColumns)
		
		return data
	
	@staticmethod
	def hashCategoricalsToInts(data):
			
		leastCommonCategoricals	= [[0]] * len(data.allColumns)

		for col in data.categoricalColumns:
			columnHist = data.rdd.map(lambda row: (row[col], 1))\
				.reduceByKey(add)\
				.sortBy(lambda pair: pair[1])
			
			leastCommonCategoricals[col] = columnHist.take(max(0, columnHist.count() - MAX_UNIQUE_CATEGORICAL_FEATURES + 2))
		
		noneHash = MAX_UNIQUE_CATEGORICAL_FEATURES #reserved hash for missing categorical features
		infrequentHash = noneHash - 1 #reserved hash for infrequent categorical values

		#performs the hashing operation on each categorical feature
		def hash(value, leastCommonCategoricalsCol):
			if value == None:
				return noneHash
			elif value in leastCommonCategoricals:
				return infrequentHash
			else:
				return int(hashlib.sha1(value).hexdigest(), 16) % (MAX_UNIQUE_CATEGORICAL_FEATURES - 2)

		data.mapOnColumnsWithParam(hash, leastCommonCategoricals, data.categoricalColumns)
		
		return data
	