import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from load_toys_rev import *

filename = 'data/toys_rev2_section3.csv'
toy_list = load_toys_rev(filename)

period = []
maxima = []
periodsum = []
count = 1
countsum = 0
for i in range(1, len(toy_list)):
    if toy_list[i].duration < toy_list[i-1].duration:
        period.append(count)
        maxima.append(toy_list[i-1].duration)
        periodsum.append(countsum + toy_list[i].duration)
        count = 1
        countsum = 0
    else:
        count += 1
        countsum += toy_list[i].duration 

plt.subplot(311)
num_bins = max(period) - min(period)
n, bins, patches = plt.hist(period, bins=num_bins, facecolor='green')
plt.xlabel('period count')
plt.ylabel('count')

plt.subplot(312)
num_bins = 100
n, bins, patches = plt.hist(maxima, num_bins, facecolor='blue')
plt.xlabel('period maximum')
plt.ylabel('count')

plt.subplot(313)
num_bins = 100
n, bins, patches = plt.hist(periodsum, num_bins, facecolor='red')
plt.xlabel('period sum')
plt.ylabel('count')

plt.show()
