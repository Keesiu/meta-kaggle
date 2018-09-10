# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:17:45 2015

@author: Team Mavericks
"""

import numpy as np

"""
Appends the probabilities for a given driver to a file
This will append "\n" first, then the actual probabilities, and then "x_" at the end
Therefore, calling appendprobabilities multiple times makes a file that is not ready for submission yet
To create a file ready for submission, call 'createsubmissionfile' after appending all the probabilities

filename: the filename of where to append the probababilities of this driver
drivernr: the number of the driver
probs: the 200-by-2 matrix with the first column containing integers (the trip nrs), 
  and the second column containing the probability that that trip belongs to the driver
fmtstring: how to format the probability. By default, this argument does not need to filled in if 
  the probability is either 0 or 1. If there are also decimals, the format string needs to be '%0.xf', 
  where x is the number of significant digits the probability needs to have
"""
def appendProbabilities(filename, drivernr, probs, fmtstring = '%0.10f'):
    #probs = np.sort(probs,axis=0)
    with open(filename, 'a') as f_handle:
        f_handle.write("\n" + str(drivernr) + "_")
        np.savetxt(f_handle, probs, header = "", footer = "", delimiter= ",", fmt=['%0.0f',fmtstring], newline = "\n" + str(drivernr) + "_", comments = "")


"""
Takes a file that is created by multiple calls of appendProbabilities, and creates a submissionfile

infilename: the filename that contains the probabilities from the 'appendProbabilities' method
outfilename: the file that needs to contain the submission file. Can be the same as the infilename,
  but in this case, all the existing content in this file will be deleted and overwritten with 
  the actual submission file
"""
def createSubmissionfile(infilename, outfilename):
    with open(infilename, 'r') as f_handle:
        data = f_handle.read()

    lines = data.split('\n')
    for i in range(len(lines)):
        if lines[i].endswith('_'):
            lines[i] = ""
        else:
            lines[i] = lines[i] + "\n"

    with open(outfilename, 'w') as f_handle:
        f_handle.write("driver_trip,prob\n")
        f_handle.write("".join(lines[1:]))    

"""
Takes a 3d matrix containing the probability matrices for every driver as slices in the 3rd dimension
This kind of 3d matrix can be created by doing np.dstack((a1,a2,...,an)) on the result
Then, creates a submission file, ready for submission to kaggle

filename: the name of the file where the submission will go. Existing contents will be overwritten
data: the 3d matrix containing the probabilities for every trip
drivernrs: for every slice of data, the driver that it belongs to. For example, data[:,:,i] belongs 
  to drivernumber drivernrs[i]. If this is not filled in, it is defaulted to data[:,:,i] belonging to
  driver number i+1 (add one because there is no driver with number 0)
fmtstring: how to format the probability. By default, this argument does not need to filled in if 
  the probability is either 0 or 1. If there are also decimals, the format string needs to be '%0.xf', 
  where x is the number of significant digits the probability needs to have
"""
def createSubmissionfileFrom3D(filename, data, drivernrs = None, fmtstring = '%0.10f'):
    if drivernrs is None:
        drivernrs = range(1,data.shape[2]+1)
    open(filename, 'w').close()    
    for i in range(data.shape[2]):
        appendProbabilities(filename, drivernrs[i], data[:,:,i], fmtstring)
    createSubmissionfile(filename, filename)
        

"""
As an example, make a submission file of three drivers, all with the same probabilities
The file "foo2" contains the preliminary data
The file "foo3.csv" contains the submission file for this example with only 3 drivers and 3 trips each
"""
def submissionfileExample():
    filename = "foo2"
    open(filename, 'w').close() #empty the file
    outfilename = "foo3.csv"
    probs = np.asarray([[3,0],[4,1],[2,1]])
    appendProbabilities(filename, 1, probs)
    appendProbabilities(filename, 2, probs)
    appendProbabilities(filename, 3, probs)
    createSubmissionfile(filename, outfilename)
    
    #Alternatively, create a 3d matrix and pass it to 'createSubmissionfileFrom3D'
    threeDimData = np.dstack((probs, probs, probs))
    createSubmissionfileFrom3D("foo4.csv", threeDimData)
