# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:46:08 2015

@author: Team Mavericks
"""
import RandomForestClassifier as learn
import readFeatureMatrix
import CreateSubmission
import numpy as np
import zipfile

"""
Reads the featurematrix, does the machine learning, creates models, uses them to predict every trip, makes submission file
input: 
  featureMatrixPath: The folder containing the features for every driver
  outputSubmissionPath: The location where the submissionfile should be generated
  trainFakeTrips: The number of fake trips to use for training the model. The feature matrix should have at least this many fake trips in it
  digits: the number of digits the submission file should have. Example, if digits = 3, the submission probablity looks like "0.123"
  foldsize: the number of trips to exclude in training, for example, if foldsize = 10, we train on 190 positive trips and then classify the other 10
  the number of folds is 200 / foldsize, so it must divide evenly. The runtime is linear in foldsize: twice as many folds means twice the runtime
output:
  importances: the importances of every feature, for every driver. This is the direct output from the random forest classifier
"""
def makeSubmissionScript(featureMatrixPath, outputSubmissionPath, trainFakeTrips = 200, digits = 5, foldsize = 10):
    #Read Feature Matrix
    featureMatrix = readFeatureMatrix.totalFeatureMatrix(featureMatrixPath)

    #ShortCut, to make loading faster, change the path if this is used.
    #featureMatrix = np.load('D:\\Documents\\Data\\MLiP\\features.npy')
    #np.save('D:\\Documents\\Data\\MLiP\\features', featureMatrix)
    
    print(np.shape(featureMatrix))

    #some features that are not very informative, so they are ignored
    featureMatrix = np.delete(featureMatrix,  [17, 39, 42, 46, 49, 52, 53, 85, 86, 87, 126, 138, 144, 148], 0 )  
    
    #Get the driver numbers as the names of the csv files in the feature matrix path
    drivernrs = readFeatureMatrix.getdrivernrs(featureMatrixPath)
    print('Done Reading Feature matrix!')
    print(np.shape(featureMatrix))
     
    numFeat, _,numDrivers = np.shape(featureMatrix)
    numTrips = 200 #The number of trips to make the submission out of, always 200
    numfolds = float(numTrips/foldsize) #the number of folds that will be made, aka the number of models per driver
    importances = np.zeros((numFeat, numDrivers)) #Storage for feature importances
    
    #The probabilities that a trip belongs to a driver
    probabilities = np.zeros((numTrips, 2, numDrivers))
    
    for i in range(numDrivers): 
        #First get the trainingtrips from the featurematrix. It is assumed that the real trips come first
        trainTrips = np.transpose(featureMatrix[:,:(numTrips+trainFakeTrips),i])
        realTrips = trainTrips[:numTrips,:]
        trainLabels = np.hstack((np.ones(numTrips-foldsize), np.zeros(trainFakeTrips)))        
     
        #Training the model for each fold
        for b, e in [(k, k+foldsize) for k in np.arange(0,numTrips,foldsize)]:
            testIndex = np.arange(b, e)
            trainIndex = np.hstack((np.arange(0, b), np.arange(e, (numTrips+trainFakeTrips))))
            
            #Train the model, then predict the classes for that model
            model = learn.trainModel(trainTrips[trainIndex], trainLabels, criterion = 'entropy', n_trees = 300, n_jobs = -1) 
            #Keep track of the importance of each feature for this driver
            importances[:,i] = importances[:,i] +  np.divide(model.feature_importances_, numfolds)
        
            tempprobs = learn.predictClass(model, realTrips[testIndex])
            
            #Append the tripnrs, then add to probability table
            probabilities[testIndex,:,i] = np.transpose(np.vstack((testIndex+1, tempprobs)))

        #Progress report, it's so slow that we give a report every driver
        if i%1 == 0:
            print("Done learning driver " + `i`)
        #Appending the output to a file. This is reccomended when doing a long calculation because you can halt
        #the computation and still have the results for the drivers so far, but it's not needed.
        #CreateSubmission.appendProbabilities(outputSubmissionPath, drivernrs[i], probabilities[:,:,i],'%0.' + `digits` + 'f')
    print('Done calculating probabilities!')
    
    #Makes submission file
    fmtstring = '%0.' + `digits` + 'f'
    CreateSubmission.createSubmissionfileFrom3D(outputSubmissionPath, probabilities, drivernrs = drivernrs, fmtstring = fmtstring)
    
    return importances
        
if __name__ == '__main__':
    datapath = 'D:\\Documents\\Data\\MLiP\\' #The path with the features, change on your pc
    
    featureMatrixPath = datapath + 'features1000'
    outputSubmissionPath = datapath + 'submission.csv'
    trainFakeTrips = 400 #number of fake trips, change between 200 and 1000. For random forest, 400 seems to work the best
    significantdigits = 5 #number of digits in the submission file. needs to be at least 3 for good precision
    foldsize = 10
    makeSubmissionScript(featureMatrixPath, outputSubmissionPath, trainFakeTrips, significantdigits, foldsize)
    
    #zip the submission, makes it ~3x smaller
    zf = zipfile.ZipFile(outputSubmissionPath[:-4] + '.zip', mode='w')
    zf.write(outputSubmissionPath, 'submission.csv', compress_type=zipfile.ZIP_DEFLATED)
    zf.close()
    print('Done creating submission!')
