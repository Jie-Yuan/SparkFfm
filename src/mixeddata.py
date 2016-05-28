import math
import hashlib
from pyspark import *
from operator import add
import os

from extensions import Extensions
from logger import Logger

MAX_UNIQUE_CATEGORICAL_FEATURES = long(10 ** 4)

class MixedData():
	
	def __init__(self, sc, dataPath, headerFilePath):
		
		self.sc = sc
		self.uniqueCategoricals = MAX_UNIQUE_CATEGORICAL_FEATURES
		
		#initilize information about features
		self.featureNames = open(headerFilePath, "r").read().split("\t")
		self.integerColumns = [col for col in range(len(self.featureNames)) if self.featureNames[col][0] == 'I']
		self.categoricalColumns = [col for col in range(len(self.featureNames)) if self.featureNames[col][0] == 'C']
		self.labelColumns = [col for col in range(len(self.featureNames)) if self.featureNames[col][0] == "L"]
		self.allColumns = range(len(self.featureNames))
		
		if os.path.isdir(dataPath):
			#data is folder of object files, assume preprocessed
			Logger.info('Loading from pickle file at \'' + dataPath + '\'...')
			self.rdd = sc.pickleFile(dataPath)
			self.preprocessed = True
			Logger.info('Complete.')
		else:
			#data is raw text file, needs to be parsed
			Logger.info('Parsing text file at \'' + dataPath + '\'...')
			self.rdd = sc.textFile(dataPath).map(lambda line: line.split("\t"))
			
			#convert the empty entries to None
			self.mapOnColumns(lambda value: None if value == '' else value, self.allColumns)
		
			#count the number of None entries in each column (used for integer preprocessing)
			self.noneCount = self.accumulateOnColumns(lambda value: 1 if value == None else 0, self.allColumns)

			#cast the integer features to integers
			self.mapOnColumns(lambda value: int(value) if value != None else None, self.integerColumns)
			Logger.info('Complete.')
			
			self.preprocessed = False
			
	def mapOnColumns(self, function, columns):
		
		def f(row):
			for col in columns:
				row[col] = function(row[col])
			return row
		
		self.rdd = self.rdd.map(f)
	
	def mapOnColumnsWithParam(self, function, array, columns):
		
		def f(row):
			for col in columns:
				row[col] = function(row[col], array[col])
			return row
		
		self.rdd = self.rdd.map(f)
		
	def accumulateOnColumns(self, function, columns):
		
		length = len(self.featureNames)
		accumulator = self.sc.accumulator([0] * length, Extensions.ListAccumulatorParam())

		def f(row):
			increment = [0] * length
			for col in columns:
				increment[col] = function(row[col])
			accumulator.add(increment)
			
		self.rdd.foreach(f)
		
		return accumulator.value
		
	def normaliseInts(self):

		#calculate sum of each int feature
		sum = self.accumulateOnColumns(lambda value: value, self.integerColumns)
		
		#calculate mean of each int feature
		count = self.rdd.count()
		means = [sum[col] / float(count - self.noneCount[col]) for col in self.allColumns]

		#subtract mean from each integer feature
		self.mapOnColumnsWithParam(lambda value, mean: value - mean if value != None else None, means, self.integerColumns)
		
		#calculate sum of squared difference from the mean
		squareSum = self.accumulateOnColumns(lambda value: value ** 2 if value != None else None, self.integerColumns)
		
		#calculate standard deviation of each int feature
		stdDevs = [math.sqrt(squareSum[col] / float(count - self.noneCount[col])) for col in self.allColumns]
		
		#divide each integer feature by SD
		self.mapOnColumnsWithParam(lambda value, sd: float(value) / sd if value != None and sd != 0 else value, stdDevs, self.integerColumns)

	def hashCategoricalsToInts(self):
			
		leastCommonCategoricals	= [[0]] * len(self.allColumns)

		for col in self.categoricalColumns:
			columnHist = self.rdd.map(lambda row: (row[col], 1))\
				.reduceByKey(add)\
				.sortBy(lambda pair: pair[1])
			
			leastCommonCategoricals[col] = columnHist.take(max(0, columnHist.count() - MAX_UNIQUE_CATEGORICAL_FEATURES + 2))
		
		noneHash = MAX_UNIQUE_CATEGORICAL_FEATURES - 1 #reserved hash for missing categorical features
		infrequentHash = noneHash - 1 #reserved hash for infrequent categorical values

		#performs the hashing operation on each categorical feature
		def hash(value, leastCommonCategoricalsCol):
			if value == None:
				return noneHash
			elif value in leastCommonCategoricals:
				return infrequentHash
			else:
				return int(hashlib.sha1(value).hexdigest(), 16) % (MAX_UNIQUE_CATEGORICAL_FEATURES - 2)

		self.mapOnColumnsWithParam(hash, leastCommonCategoricals, self.categoricalColumns)
		
	def preprocess(self):
		self.normaliseInts()
		self.hashCategoricalsToInts()
		self.preprocessed = True

	
		
	
	