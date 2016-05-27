from pyspark import RDD
from extensions import Extensions

class ExtendedRDD():
	
	def __init__(self, sc, dataFilePath, headerFilePath):
		
		self.sc = sc
		self.rdd = sc.textFile(dataFilePath).map(lambda line: line.split("\t"))
		self.featureNames = open(headerFilePath, "r").read().split("\t")

		self.integerColumns = [col for col in range(len(self.featureNames)) if self.featureNames[col][0] == 'I']
		self.categoricalColumns = [col for col in range(len(self.featureNames)) if self.featureNames[col][0] == 'C']
		self.labelColumns = [col for col in range(len(self.featureNames)) if self.featureNames[col][0] == "L"]
		self.allColumns = range(len(self.featureNames))
		
		#convert the empty entries to None
		self.mapOnColumns(lambda value: None if value == '' else value, self.allColumns)
		
		#count the number of None entries in each column (used for integer preprocessing)
		self.noneCount = self.accumulateOnColumns(lambda value: 1 if value == None else 0, self.allColumns)

		print("none count: " + str(self.noneCount))
		
		#cast the integer features to integers
		self.mapOnColumns(lambda value: int(value) if value != None else None, self.integerColumns)

	
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

	