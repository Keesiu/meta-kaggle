import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import copy

from load_toys_rev import *

filename = 'data/toys_rev2_sample10.csv'
sample_factor = 10
toy_list = load_toys_rev(filename)

count = [0]*3000
timesum = [0]*3000
for toy in toy_list:
    if toy.duration < 3000:
        timesum[toy.duration] += toy.duration*sample_factor

overall_timesum = []
accumulate_timesum = 0
for overall_duration in timesum:
    accumulate_timesum += overall_duration
    overall_timesum.append(accumulate_timesum)

f = open('overall_duration_sum.csv', 'w')
i = 0
for overall in overall_timesum:
    f.write(str(i)+", "+str(overall)+"\n")
    i += 1
f.close()

plt.plot(overall_timesum, 'r')
plt.xlabel('duration (min)')
plt.ylabel('overall duration sum (min)')

plt.show()
