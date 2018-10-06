# -*- coding: utf-8 -*-

import pandas
import numpy as np
import random
from sklearn.externals import joblib
from sklearn.preprocessing import scale
from sklearn import svm

csv = np.array(('driver_trip', 0.0))

features = joblib.load('features77.pkl')
features = scale(features)
drivernames = joblib.load('drivernames.pkl')
countdriver = len(drivernames)

countrandom = 200
countiter = 5

data = np.empty((countrandom+200,features.shape[1]))
label = np.zeros(countrandom+200) 
clf = svm.SVC(probability=True, cache_size=1000)
         
label[0:200] = 1
       
#for every driver
for driver in range(countdriver):
    
    #prepare data
    data[0:200,:] = features[driver*200:(driver+1)*200,:]
    
    prob = np.zeros(200)
    
    #make first column of result    
    names = []
    for route in range(1,201):
        names.append(str(int(drivernames[driver])) + '_' + str(route))
        
    for currentiter in range(countiter):
        
        #prepare data    
        for iter in range(countrandom):
            randomdriver = random.randint(0, countdriver-1)
            while driver == randomdriver:
                randomdriver = random.randint(0, countdriver-1)
                
            randomroute = random.randint(0, 199)
            data[200+iter,:] = features[randomdriver*200+randomroute,:]
          
        #svm
        clf.fit(data, label)
        prob += clf.predict_proba(data[0:200,:])[:,1]
        
    prob /= countiter
        
    probstr = []
    for iter in prob:
        probstr.append('%.6f' % iter)    
    
    csv = np.vstack((csv, np.column_stack((names, probstr))))
    
#saving csv
df = pandas.DataFrame(csv)
df.to_csv('resultsvm.csv')

