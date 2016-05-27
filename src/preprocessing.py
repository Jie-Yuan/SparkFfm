import math
import hashlib
from pyspark import *

from extensions import Extensions

MAX_UNIQUE_CATEGORICAL_FEATURES = long(10 ** 4)

class PreProcessing:
	@staticmethod
	def normaliseInts(sc, rdd, integerFeatures, noneCount):
		integerColumns = [feature.column for feature in integerFeatures]
		
		#calculate sum of each int feature
		sumAccum = sc.accumulator([0] * len(integerColumns), Extensions.ListAccumulatorParam())
		rdd.foreach(lambda row: sumAccum.add([row[i] for i in integerColumns]))
		
		#calculate mean of each int feature
		count = rdd.count()
		sum = sumAccum.value
		
		print("sum: " + str(sum))
		print("noneCount: " + str(noneCount))
		
		means = [sum[i] / float(count - noneCount[i]) for i in range(len(sum))]
		
		print("means: " + str(means))
		
		#subtract mean from each integer feature
		def subtractMeanFromIntColumns(row):
			for i in range(len(integerColumns)):
				if row[integerColumns[i]] != None:
					row[integerColumns[i]] -= means[i]
			return row
		
		rdd = rdd.map(subtractMeanFromIntColumns)
		
		#calculate sum of squared difference from the mean
		squareSumAccum = sc.accumulator([0] * len(integerColumns), Extensions.ListAccumulatorParam())
		rdd.foreach(lambda row: squareSumAccum.add([row[i] ** 2 if row[i] != None else None for i in integerColumns]))
		
		#calculate standard deviation of each int feature
		squareSum = squareSumAccum.value
		sd = [math.sqrt(squareSum[i] / float(count - noneCount[i])) for i in range(len(squareSum))]
		
		#divide each integer feature by SD
		def divideIntColumnsBySD(row):
			for i in range(len(integerColumns)):
				if sd[i] != 0 and row[integerColumns[i]] != None:
					row[integerColumns[i]] = row[integerColumns[i]] / sd[i]
			return row
		
		rdd = rdd.map(divideIntColumnsBySD)
		
		return rdd
		
	@staticmethod
	def hashCategoricalsToInts(rdd, categoricalFeatures):
		missingHash = MAX_UNIQUE_CATEGORICAL_FEATURES #reserved hash for missing categorical features
		categoricalColumns = [feature.column for feature in categoricalFeatures]
		
		#performs the hashing operation on each categorical feature
		def doHash(row):
			for i in range(len(categoricalColumns)):
				feature = row[categoricalColumns[i]]
				if feature == '':
					row[categoricalColumns[i]] = missingHash
				else:
					row[categoricalColumns[i]] = int(hashlib.sha1(feature).hexdigest(), 16) % (MAX_UNIQUE_CATEGORICAL_FEATURES - 1)
			return row
		
		rdd = rdd.map(doHash)
		
		return rdd
	