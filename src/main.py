from pyspark import *
from pyspark.sql import *
from operator import add
import math
import hashlib

MAX_UNIQUE_CATEGORICAL_FEATURES = long(10 ** 4)

class ArgumentException(Exception):
	pass
	
class Feature:
	
	class Type:
		LABEL = 0
		INTEGER = 1
		CATEGORICAL = 2
		
	def __init__(self, string, col):
		self.column = col
		if string[0] == 'L':
			self.type = Feature.Type.LABEL
			pass			
		elif string[0] == 'I':
			self.type = Feature.Type.INTEGER
		elif string[0] == 'C':
			self.type = Feature.Type.CATEGORICAL
		else:
			raise ArgumentException("Unrecognised feature identifier:" + string)
		
		if self.type != Feature.Type.LABEL:
			self.index = int(string[1:])
		
	def toString(self):
		if self.type == Feature.Type.LABEL:
			return 'L'
		elif self.type == Feature.Type.INTEGER:
			return 'I' + str(self.index)
		else:
			return 'C' + str(self.index)
		
class ListAccumulatorParam(AccumulatorParam):
	def zero(self, initialValue):
		return [0] * len(initialValue)
		
	def addInPlace(self, a, b):
		#None-safe addition (None treated as zero)
		a = [0 if x == None else x for x in a]
		b = [0 if x == None else x for x in b]
		return map(add, a, b)
		
class PreProcessing:
	@staticmethod
	def normaliseInts(rdd, integerFeatures, noneCount):
		integerColumns = [feature.column for feature in integerFeatures]
		
		#calculate sum of each int feature
		sumAccum = sc.accumulator([0] * len(integerColumns), ListAccumulatorParam())
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
		squareSumAccum = sc.accumulator([0] * len(integerColumns), ListAccumulatorParam())
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
		
###MAIN###
		
sc = SparkContext("local", "Categorical Histograms")
sql = SQLContext(sc)

#TODO: if requiresPreprocessing
dataFilePath = "data/dac_sample.txt"
headerFilePath = "data/dac_header.txt"
preprocessedRddPath = "data/preprocessed"
featureNames = open(headerFilePath, "r").read().split("\t")
rdd = sc.textFile(dataFilePath).map(lambda line: line.split("\t"))

#rdd = rdd.sample(False, 0.0001)

features = [Feature(featureNames[col], col) for col in range(len(featureNames))]
integerFeatures = [feature for feature in features if feature.type == Feature.Type.INTEGER]
categoricalFeatures = [feature for feature in features if feature.type == Feature.Type.CATEGORICAL]

integerColumns = [feature.column for feature in integerFeatures]
intNoneCount = sc.accumulator([0] * len(integerColumns), ListAccumulatorParam())
#casts RDD elements to ints if they are integer type
def castInts(row):
	for i in range(len(integerColumns)):
		if row[integerColumns[i]] == '':
			row[integerColumns[i]] = None
			increment = [0] * len(integerColumns)
			increment[i] = 1
			intNoneCount.add(increment)
		else:
			row[integerColumns[i]] = int(row[integerColumns[i]])
	return row

rdd = rdd.map(castInts)
rdd.collect()

print("intNoneCount: " + str(intNoneCount.value))

rdd = PreProcessing.normaliseInts(rdd, integerFeatures, intNoneCount.value)
rdd = PreProcessing.hashCategoricalsToInts(rdd, categoricalFeatures)

print(rdd.take(1))

rdd.collect()
rdd.saveAsTextFile(preprocessedRddPath)


			
	
