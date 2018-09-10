import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from load_toys_rev import *

filename = 'data/toys_rev2_sample10.csv'
sample_factor = 10
toy_list = load_toys_rev(filename)

count = [0]*3000
timesum = [0]*3000
for toy in toy_list:
    if toy.duration < 3000:
        count[toy.duration] += 1*sample_factor
        timesum[toy.duration] += toy.duration*sample_factor

plt.subplot(211)
plt.semilogy(count, 'r')
plt.xlabel('duration (min)')
plt.ylabel('count')

plt.subplot(212)
plt.semilogy(timesum, 'b')
plt.xlabel('duration (min)')
plt.ylabel('time sum (min)')

plt.show()
