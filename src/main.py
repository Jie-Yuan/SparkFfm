from pyspark import *
import math

#sparkFFM
from preprocessing import PreProcessing
from extensions import Extensions
	
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
		

		
	
###MAIN###
		
sc = SparkContext("local", "Main")

#TODO: if requiresPreprocessing
dataFilePath = "data/dac_sample.txt"
headerFilePath = "data/dac_header.txt"
preprocessedTextPath = "data/preprocessed-text"
preprocessedSerPath = "data/preprocessed-ser"
featureNames = open(headerFilePath, "r").read().split("\t")
rdd = sc.textFile(dataFilePath).map(lambda line: line.split("\t"))

#rdd = rdd.sample(False, 0.0001)

features = [Feature(featureNames[col], col) for col in range(len(featureNames))]
integerFeatures = [feature for feature in features if feature.type == Feature.Type.INTEGER]
categoricalFeatures = [feature for feature in features if feature.type == Feature.Type.CATEGORICAL]

integerColumns = [feature.column for feature in integerFeatures]
intNoneCount = sc.accumulator([0] * len(integerColumns), Extensions.ListAccumulatorParam())
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

rdd = PreProcessing.normaliseInts(sc, rdd, integerFeatures, intNoneCount.value)
rdd = PreProcessing.hashCategoricalsToInts(rdd, categoricalFeatures)

print(rdd.take(1))

rdd.collect()
rdd.saveAsTextFile(preprocessedTextPath)
rdd.saveAsPickleFile(preprocessedSerPath)


			
	
