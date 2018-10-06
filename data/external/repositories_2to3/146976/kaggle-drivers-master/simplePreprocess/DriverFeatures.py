# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:14:22 2015

@author: Team Mavericks
"""

import numpy as np
from scipy.stats import percentileofscore

"""
Given two trips, computes the euclidean distance between them
It also flips one of the trips, to find a better match between them
There is a multiplier to penalize short trips, because the euclidean distance 
between short (100m) trips is always very low, so you would always find a match
"""
def matchTrips(t, s):
    distmult = min(t.cumdist[-1], s.cumdist[-1])**2.0
    matchone = np.sum(np.hypot(t.newX-s.newX, t.newY-s.newY))
    matchtwo = np.sum(np.hypot(t.newX-s.newX, t.newX+s.newY))
    return np.min((matchone, matchtwo)) / distmult

"""
Given a list of trips, and a number, finds the best match for the trip indexed by that number
It does this by checking all other trips, and returning the distance between the trip and its best match
"""
def tripMatch(triplist, tripnr):
    bestmatch = -1
    numtrips = len(triplist)
    for i in range(numtrips):
        if i == tripnr or (tripnr == -1 and i == numtrips - 1):
            continue
        m = matchTrips(triplist[i], triplist[tripnr])
        if m < bestmatch or bestmatch == -1:
            bestmatch = m
    return bestmatch

"""
Given a list of trips and a number of a trip, the name of an attribute, and what operation to do on that attribute,
returns the percentile of that trip in the full list of drivers.
To be more clear, the operation should take in the attribute and give out a scalar
Then, these scalars are compared, and the percentile of that scalar of the given trip is returned.
"""
def getPercentile(triplist, tripnr, attrname, op=np.mean):
    return percentileofscore([op(trip.__dict__[attrname]) for trip in triplist], op(triplist[tripnr].__dict__[attrname]))

maxVelPerc = lambda triplist, tripnr: getPercentile(triplist, tripnr,'dist', np.max)
maxAccPerc = lambda triplist, tripnr: getPercentile(triplist, tripnr,'anorm', np.max)
minAccPerc = lambda triplist, tripnr: getPercentile(triplist, tripnr,'anorm', np.min)
maxStePerc = lambda triplist, tripnr: getPercentile(triplist, tripnr,'snorm', np.max)
minStePerc = lambda triplist, tripnr: getPercentile(triplist, tripnr,'snorm', np.min)

getVelocityPercentile = lambda triplist, tripnr: getPercentile(triplist, tripnr,'v')
getAccelerationPercentile = lambda triplist, tripnr: getPercentile(triplist, tripnr, 'a')

"""
Count Feature
Counts all nonzero (or true) elements, and returns the number of them
by default, normalizes it, so it returns the proportion of elements that are nonzero (or true)
"""
def coF(array, normalize = True):
    count = float(np.count_nonzero(array))
    if normalize:
        count = count / len(array)
    return count

"""
Counts distance
counts distance for all nonzero (or true) elements, and returns the sum of them
by default, normalizes it, so it returns the proportion of the trip that is nonzero (or true)
"""
def coD(array, dist, normalize=True):
    distInInterval = np.sum(dist[array])
    if normalize:
        distInInterval = distInInterval / float(np.sum(dist))
    return distInInterval

"""
Gives the proportion of elements that are in some interval.
The interval is given by (lowval, highval), and we look at an attribute given by attrname
For example, if attrname = 'v', we look at the proportion of elements that have a velocity in (lowval, highval)
"""
def proportionInInterval(triplist, tripnr, attrname, lowval, highval):
    tripvalues = triplist[tripnr].__dict__[attrname]
    result = coF((tripvalues >= lowval) & (tripvalues <= highval))
    #print (lowval, highval, result)
    return result

"""
Gives the proportion of the trip (distance-wise) that is in some interval.
The interval is given by (lowval, highval), and we look at an attribute given by attrname
For example, if attrname = 'v', we look at the proportion of the trip that has a velocity in (lowval, highval)
"""
def propDistInInterval(triplist, tripnr, attrname, lowval, highval):
    tripvalues = triplist[tripnr].__dict__[attrname]
    result = coD((tripvalues >= lowval) & (tripvalues <= highval), triplist[tripnr].dist)
    #print (lowval, highval, result)
    return result

"""
Given a list of trips, an attribute and two percentiles, 
computes the value belonging to those percentiles
For example, if attrname = 'v', lowperc = 5, highperc = 95, 
this function returns the 5th and 95th percentiles of velocity for all the trips 
in triplist
"""
def getPercentiles(triplist, attrname, lowperc, highperc):
    attr = np.hstack([trip.__dict__[attrname] for trip in triplist])
    lowval = np.percentile(attr, lowperc)
    highval = np.percentile(attr, highperc)
    return lowval, highval

propVel10 = ('t','v', 0, 10)
propVel30 = ('t', 'v', 10,30)
propVel70 = ('t', 'v', 30,70)
propVel90 = ('t', 'v', 70,90)
propVel100 = ('t', 'v', 90, 100)
propAcc10 = ('t', 'a', 0, 10)
propAcc30 = ('t', 'a', 10,30)
propAcc70 = ('t', 'a', 30,70)
propAcc90 = ('t', 'a', 70,90)
propAcc100 = ('t', 'a', 90, 100)

propDVel10 = ('d', 'v', 0, 10)
propDVel30 = ('d', 'v', 10,30)
propDVel70 = ('d', 'v', 30,70)
propDVel90 = ('d', 'v', 70,90)
propDVel100 = ('d', 'v', 90, 100)
propDAcc10 = ('d', 'a', 0, 10)
propDAcc30 = ('d', 'a', 10,30)
propDAcc70 = ('d', 'a', 30,70)
propDAcc90 = ('d', 'a', 70,90)
propDAcc100 = ('d', 'a', 90, 100)

"""
Given a list of real trips and fake trips and a feature in the form of a tuple, 
computes that feature for all the trips, and returns it as an array
The feature looks like (method, attrname, lowperc, highperc)
this corresponds to a feature that computes the 'method' where attrname is in the interval (lowperc, highperc)
Basically, it returns proportionInInterval or propDistInInterval, so see those functions for more information
"""
def makeDriverFeaturePercentile(realTrips, fakeTrips, feature):
    prop, attrname, lowperc, highperc = feature
    lenR = len(realTrips)
    lenF = len(fakeTrips)
    result = np.zeros(lenR + lenF)
    attr = np.hstack([trip.__dict__[attrname] for trip in realTrips])
    lowval = np.percentile(attr, lowperc)
    highval = np.percentile(attr, highperc)    
    del attr
    for i in range(lenR):
        if prop == 'd':
            result[i] = propDistInInterval(realTrips, i, attrname, lowval, highval)
        if prop == 't':
            result[i] = proportionInInterval(realTrips, i, attrname, lowval, highval)
    realTrips.append(0)
    for i in range(lenF):
         realTrips[-1] = fakeTrips[i]
         if prop == 'd':
             result[lenR+i] = propDistInInterval(realTrips, -1, attrname, lowval, highval)
         if prop == 't':
             result[lenR+i] = proportionInInterval(realTrips, -1, attrname, lowval, highval)
    del realTrips[-1]
    return result

"""
Given some amount of real trips, and some amount of fake trips, and a driver feature, return
an array with that feature for all the trips. The feature should be a function that takes in a list of trips,
and a trip nr, and gives out a scalar. The feature can also be a tuple, in which case, this function defers to
'makeDriverFeaturePercentile'

It computes the feature by comparing the 200 real trips to each other, then comparing each fake trip individually to
all the real trips. The fake trips are never compared to each other, which is the goal of the driver features.
"""
def makeDriverFeature(realTrips, fakeTrips, feature):
    if isinstance(feature, tuple):
        return makeDriverFeaturePercentile(realTrips, fakeTrips, feature)
    lenR = len(realTrips)
    lenF = len(fakeTrips)
    result = np.zeros(lenR + lenF)
    for i in range(lenR):
        result[i] = feature(realTrips, i)
    realTrips.append(0)
    for i in range(lenF):
        realTrips[-1] = fakeTrips[i]
        result[i+lenR] = feature(realTrips, -1)
    del realTrips[-1]
    return result


driverfeatures = [tripMatch,
                  getVelocityPercentile,
                  getAccelerationPercentile,
                  propVel10,
                  propVel30,
                  propVel70,
                  propVel90,
                  propVel100,
                  propAcc10,
                  propAcc30,
                  propAcc70,
                  propAcc90,
                  propAcc100,
                  propDVel10,
                  propDVel30,
                  propDVel70,
                  propDVel90,
                  propDVel100,
                  propDAcc10,
                  propDAcc30,
                  propDAcc70,
                  propDAcc90,
                  propDAcc100,
                  maxVelPerc,
                  maxAccPerc,
                  minAccPerc,
                  maxStePerc,
                  minStePerc ]
                  
if __name__ == '__main__':
    driver = 'D:\Documents\Data\MLiP\drivers\\1\\'
    fakedriver = 'D:\Documents\Data\MLiP\drivers\\2\\'

    realtripfiles = [driver + repr(i) + '.csv' for i in range(1,101)]
    faketripfiles = [driver + repr(i) + '.csv' for i in range(101,201)]
    
    from .simplePyprocessor import trip
    realtrips = []
    faketrips = []
            
    for i, file in enumerate(realtripfiles):
        t = trip(file)
        realtrips.append(t)
        
    faketrips = []
    for i, file in enumerate(faketripfiles):
        t = trip(file)
        faketrips.append(t)
        
    print((makeDriverFeature(realtrips, faketrips, propDVel30)))
