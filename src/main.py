from pyspark import *
import math

#sparkFFM
from extendedrdd import ExtendedRDD
from preprocessing import PreProcessing
from extensions import Extensions

###MAIN###
		
sc = SparkContext("local", "Main")

#TODO: if requiresPreprocessing
dataFilePath = "data/dac_sample.txt"
headerFilePath = "data/dac_header.txt"
preprocessedTextPath = "data/preprocessed-text"
preprocessedSerPath = "data/preprocessed-ser"

data = ExtendedRDD(sc, dataFilePath, headerFilePath)
data.rdd.persist()

print(data.rdd.take(1))

data = PreProcessing.normaliseInts(data)

print(data.rdd.take(1))

data = PreProcessing.hashCategoricalsToInts(data)

print(data.rdd.take(1))

data.rdd.saveAsTextFile(preprocessedTextPath)
data.rdd.saveAsPickleFile(preprocessedSerPath)


			
	
