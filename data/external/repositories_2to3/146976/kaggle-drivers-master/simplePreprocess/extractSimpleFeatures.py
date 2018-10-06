import numpy as np

from .simplePyprocessor import trip
from .simpleFeatures import features
from .driverFeatures import driverfeatures, makeDriverFeature
from os import listdir, path

full_data = '../data/drivers'

"""
Gets the drivernrs from the data, used for file I/O
input: tdatpath is the path to the raw data from kaggle, with foldernames equal to the drivernrs
"""
def getdrivernrs(tdatpath):
    files = listdir(tdatpath)
    return np.array(sorted([int(f) for f in files]))

"""
Creates the featurematrix for a single driver, and writes it to output
input: driver_id, the id of the driver of which we have to create the feature matrix
numT: The number of randomly chosen fake trips that will be included in this featureMatrix
"""
def main(driver_id, numT = 200):
    folder = '%s/%s'%(full_data, driver_id)
    tripfiles = ['%s/%s'% (folder, f) for f in listdir(folder)]
    drivernrs = getdrivernrs(full_data)
    numR = len(tripfiles)
    simF = len(features) #simple features
    advF = len(driverfeatures) #driver features

    #The feature Matrix
    feats = np.empty((numR + numT, simF + advF))

    #The following 5 lines create the files of the real driver, to read later on
    #The order of the files matter, so we sort them numerically
    driverfolder = path.dirname(tripfiles[0]) + '/'
    basenames = [path.basename(f) for f in tripfiles]
    basenames = [int(f[:-4]) for f in basenames]
    basenames =  [str(f) + '.csv' for f in sorted(basenames)]
    realtripfiles = [driverfolder + b for b in basenames]
    
    #We select some random drivers and random trips
    #We also set the probablity of selecting the driver with driver_id equal to 0,
    #because that is not fake trip anymore.
    driverprobs = np.ones(len(drivernrs))
    driverprobs[drivernrs == driver_id] = 0
    driverprobs = driverprobs / float(len(drivernrs) - 1)
    randD = np.random.choice(drivernrs, size = numT ,replace=False, p = driverprobs)
    randT = np.random.randint(1 ,numT+1, size = numT) 
    fakefolder = ['%s/%s'%(full_data, did) for did in randD]
    faketripfiles = ['%s/%s.csv'%(fakefolder[i], randT[i]) for i in range(numT)]
    #we now have the paths of the fake tripfiles and the real tripfiles
    
    #We make a list of real trips, and immediately compute the simple features for each trip
    realtrips = []
    for i, file in enumerate(realtripfiles):
        t = trip(file)
        realtrips.append(t)
        feats[i,:simF] = np.array([f(t) for f in features])
    
    #We make a list of fake trips, and immediately compute the simple features for each trip
    faketrips = []
    for i, file in enumerate(faketripfiles):
        t = trip(file)
        faketrips.append(t)
        feats[i+numR,:simF] = np.array([f(t) for f in features])
    
    #Now that we have read all the trips, we compute each driver feature, one by one
    for i, f in enumerate(driverfeatures):
        feats[:,simF + i] = makeDriverFeature(realtrips, faketrips, f)
    
    #We write the feature matrix to output
    write(driver_id, feats)
    print(driver_id)

"""
creating the output file as simply as possible.
This contains one trip per row, and one feature per column
The outputfile is placed in ./output/###.csv, where ### is the drivernr
"""
def write(driver, feats):
    from csv import writer
    with open('./output/%s.csv'% driver,'w') as outfile:
        writer(outfile).writerows(feats)

if __name__=='__main__':
    #We get as an input argument the driver nr, and we make the featurematrix of that driver
    from sys import argv

    folder = argv[1]
    main(folder, 400) #The number of fake trips. Runtime increases dramatically with this, because
                      #The driverfeatures mean that each fake trip has to be compared with 200 real trips


#sample usage:
#This calls this file in parallel for every driver, then zips the output
#Warning: for more than 100 features, the output will be several GB's  
#parallel -j 24 python ./extractSimpleFeatures.py -- `ls ../data/drivers`
#cd output
#zip -r ../features.zip *
#cd ..
