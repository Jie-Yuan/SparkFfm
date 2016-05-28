from pyspark import *
import math
import os
import shutil
import sys
import numpy

#sparkFFM
from mixeddata import MixedData
from preprocessing import PreProcessing
from extensions import Extensions
from logger import Logger
from model import Model

###MAIN###
		
sc = SparkContext("local", "Main")

#TODO: if requiresPreprocessing
textFilePath = "data/dac_sample.txt"
headerFilePath = "data/dac_header.txt"
preprocessedTextPath = "data/preprocessed-text"
preprocessedSerPath = "data/preprocessed-ser"
weightsPath = "data/weights"

usePPD = False
if '-usePPD' in sys.argv:
	if os.path.exists(preprocessedSerPath):
		usePPD = True
	else:
		Logger.warn('No pre-processed data exists. Using raw data instead.')

if usePPD:
	data = MixedData(sc, preprocessedSerPath, headerFilePath)
else:
	data = MixedData(sc, textFilePath, headerFilePath)
	
	Logger.info('Preprocessing data...')
	data.preprocess()
	Logger.info('Complete.')
	
	Logger.info('Deleting any existing pre-processed data...')
	for path in [preprocessedTextPath, preprocessedSerPath]:
		shutil.rmtree(path, ignore_errors = True)
	Logger.info('Complete.')
	
	Logger.info('Saving pre-processed data to disk...')
	data.rdd.persist()
	data.rdd.saveAsTextFile(preprocessedTextPath)
	data.rdd.saveAsPickleFile(preprocessedSerPath)
	Logger.info('Complete.')

model = Model(data)

model.stochasticGradientDescent(data)

numpy.savetxt(weightsPath + '/integer', model.weights.integer)
numpy.savetxt(weightsPath + '/categorical', model.weights.categorical)

			
	
