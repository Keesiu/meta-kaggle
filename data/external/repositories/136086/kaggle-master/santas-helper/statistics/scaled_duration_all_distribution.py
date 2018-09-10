import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from load_toys_rev import *

filename = 'data/toys_rev2_sample10.csv'
sample_factor = 10
toy_list = load_toys_rev(filename)

def scale (duration):
    if duration < 150:
        return duration*4
    if duration <= 2400:
        return 600
    else:
        return duration*0.25

count = [0]*3000
timesum = [0]*3000
for toy in toy_list:
    if toy.duration < 3000:
        count[toy.duration] += 1*sample_factor
        timesum[toy.duration] += scale(toy.duration)*sample_factor

plt.subplot(211)
plt.semilogy(count, 'r')
plt.xlabel('duration (min)')
plt.ylabel('count')

plt.subplot(212)
plt.semilogy(timesum, 'b')
plt.xlabel('duration (min)')
plt.ylabel('scaled time sum (min)')

plt.show()
