# -*- coding: utf-8 -*-

import pandas
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

csv = np.array((0,2))
features = joblib.load('features77.pkl')
drivernames = joblib.load('drivernames.pkl')
countdriver = len(drivernames)

countrandom = 800
countiter = 4

data = np.empty((countrandom+200,features.shape[1]))
label = np.zeros(countrandom+200)
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
        
        #randomforest
        forest = RandomForestClassifier(criterion='entropy', n_estimators = 2000, n_jobs = -1) 
        forest.fit(data, label)  
        
        prob += forest.predict_proba(data[0:200,:])[:,1]

    #mean of 5 iteration    
    prob /= countiter
    
    #adding to result
    csv = np.vstack((csv, np.column_stack((names, prob))))
    
#saving csv
df = pandas.DataFrame(csv)
df.to_csv('result.csv')

