# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:12:33 2015

@author: Team Mavericks
"""
import os
import numpy as np

"""
A string sorting method, that sorts csv files not alphabetically, but numerically
Sorts [1.csv, 10.csv, 2.csv, ...] as [1.csv, 2.csv, ...]
This is used because os.listdir sorts alphabetically so we need to sort it correctly.
"""
def sortNumerical(filelist):
    filelist = [int(f[:-4]) for f in filelist]
    return [repr(f) + '.csv' for f in sorted(filelist)]

"""
Get an array of integers, corresponding to the driver numbers
This is because the driver numbers to not run nicely from 1 to 2736,
but instead skip some intermediate numbers
The input is a folder with csv files, with the name of the csv files equal to the driver number
This is typically the feature matrix path
"""
def getdrivernrs(tdatpath):
    files = os.listdir(tdatpath)
    files = sortNumerical(files)
    return np.array([int(f[:-4]) for f in files])

"""
Given a feature matrix path, we create a feature matrix by putting together all the individual csv files
input is the featrure matrix path, the size of the matrix, and if we should fix nans or not
The size of the matrix can be computed with "getNumFeatures", "getNumTrips", and "len(getdrivernrs)"
Warning: this function does a lot of I/O, so it can take a while
"""
def makeFeatureMatrix(tdatpath, numFeatures, numTrips, numDrivers, fixnan = True):
    featurematrix = np.zeros((numFeatures, numTrips,numDrivers))
    files = os.listdir(tdatpath)
    files = sortNumerical(files)
    for i in range(numDrivers):
        csvpath = os.path.join(tdatpath, files[i])
        features = np.transpose(np.genfromtxt(csvpath, dtype = 'float', delimiter = ','))
        if fixnan:
            features = np.nan_to_num(features)
        featurematrix[:,:,i] = features
    return featurematrix

"""
Gets the number of features from a csv file with features
Basically, it counts the number of commas, since that is the delimiter
""" 
def getNumFeatures(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        return first_line.count(',') + 1
   
"""
Gets the number of trips from a csv file with features
Simply counts the number of lines
"""
def getNumTrips(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

"""
Makes the total feature matrix, with only a single input argument
Does this by calculating the number of features, the number of drivers, and the number of trips
Then it calls the 'makeFeatureMatrix' function
"""
def totalFeatureMatrix(featureMatrixPath):
    numDrivers = len(os.listdir(featureMatrixPath))
    firstdriver = os.path.join(featureMatrixPath, os.listdir(featureMatrixPath)[0])
    numTrips = getNumTrips(firstdriver)
    numFeatures = getNumFeatures(firstdriver)
    featureMatrix = makeFeatureMatrix(featureMatrixPath, numFeatures, numTrips, numDrivers)	
    return featureMatrix	


if __name__ == '__main__':
    print(np.__version__)    
    tdatpath = 'D:\Documents\Data\MLiP\output'
    numFeatures = 6
    numTrips = 200
    numDrivers = 10
    print((makeFeatureMatrix(tdatpath, numFeatures, numTrips, 10)))
