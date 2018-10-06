#-*- coding: utf-8 -*-
"""
Created on Sat Mar 07 14:22:35 2015

@author: Team Mavericks
"""

import numpy as np
from csv import reader
from os import listdir
from scipy.fftpack import fft

"""
Sets a trip to be a fixed number of datapoints
Used to compare two trips to each other
It does this by dividing the trips into a number of pieces, lengthwise,
and placing datapoints among these even division
When the trip is not actually 1500 datapoints, the resulting array will have zeros at the end
"""	
def tripFixedLength(x, y, cumdist, datapoints = 1500):
    lperd = cumdist[-1] / float(datapoints-1)
    newx = np.zeros(datapoints)
    newy = np.zeros(datapoints)
    fill = 1
    for i in range(len(cumdist)):
        if cumdist[i] > fill * lperd:
            newx[fill] = x[i]
            newy[fill] = y[i]
            fill = fill + 1
    newx[-1] = x[-1]
    newy[-1] = y[-1]
    return newx, newy

"""
Class used to store all kinds of information about a trip, like
velocity, acceleration, etc. This is calculated once, so that not every single feature has to 
recompute the velocity. That is why this class only contains the basic information, 
the actual handcrafted features do the rest of the work
"""
class trip(np.ndarray):

    def __new__(cls, filename, precision=1, **kwargs):
        with open(filename) as tripfile:
            head = tripfile.readline()
            trip = np.array(list(reader(tripfile)), dtype=float)
        return np.round(trip, decimals=precision).view(cls)

    def __init__(self, filename, **kwargs):

        X, Y = self.T
        self.n = self.shape[0] - 1 #length of trip
        self.t = np.arange(self.n) 
        self.x = X #raw data
        self.y = Y
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.ddx = np.diff(self.dx)
        self.ddy = np.diff(self.dy)
        self.ddx = np.hstack((0, self.ddx))
        self.ddy = np.hstack((0, self.ddy))

        #The actual features
        self.v = np.hypot(self.dx, self.dy) #velocity
        self.v = np.hstack((self.v[0], self.v))
        self.o = np.arctan2(self.dy, self.dx) #orientation, or facing direction
        self.s = np.diff(self.o) #steering, difference in orientation
        self.s = np.hstack((self.s[0], self.s, self.s[-1]))
        self.a = np.diff(self.v) #Acceleration
        self.a = np.hstack((self.a[0], self.a))
        self.ds = np.diff(self.s)
        
        #polar coordinates
        self.rad = np.hypot(self.x, self.y)
        self.phi = np.arctan2(self.y, self.x)
        meanphi = np.mean(self.phi)
        self.normphi = self.phi - meanphi #normalized trip, by setting the average angle to 0
        self.normX = self.rad * np.cos(self.normphi)
        self.normY = self.rad * np.sin(self.normphi)
        
        self.dist = np.copy(self.v) #The distance traveled at each point, by taking out hyperjumps
                                    #A hyperjump is defined as going faster than 180 km/h
        self.dist[self.dist > 50] = 0
        self.cumdist = np.cumsum(self.dist)
        self.straightdist = np.hypot(self.x, self.y)
        self.anorm = np.copy(self.a)
        self.anorm[self.dist > 50] = 0
        self.snorm = np.copy(self.s)
        self.snorm[self.dist > 50] = 0
        
        #the trip by setting it to 1500 datapoints
        self.newX, self.newY = tripFixedLength(self.normX, self.normY, self.cumdist)

if __name__ == '__main__':
#examples:
    total_time = lambda trip: trip.n
    total_distance = lambda trip: np.sum(np.hypot(np.diff(trip[:,0]), np.diff(trip[:,1])))
    straight_distance = lambda trip: np.hypot(trip[-1,0], trip[-1,1])
    straightness = lambda trip: straight_distance(trip) / total_distance(trip)
    sum_turnspeeds = lambda trip: np.sum(trip.s/trip.v)**2
    acceleration_to_dist = lambda trip: np.sum(trip.a**2)

    total_standstill_time = lambda trip: np.count_nonzero(trip.v < 0.1)

    trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'
    tripp = trip(trippath)

    print((tripp.v))
    print((total_standstill_time(tripp)))
