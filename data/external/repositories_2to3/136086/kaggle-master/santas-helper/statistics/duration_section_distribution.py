import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from load_toys_rev import *

filename = 'data/toys_rev2_sample113.csv'
sample_factor = 113
toy_list = load_toys_rev(filename)

cut1 = 2600*800
cut2 = 6500000 # confirmed
cut3 = 8750000 # confirmed

plt.subplot(221)
num_bins = 60
n, bins, patches = plt.hist(list(
    [x for x in [x.duration for x in toy_list[:int(cut1/sample_factor)]] if x < 10*60]),
    num_bins, facecolor='green')
#count = mlab.normpdf(bins, 100, 15)
#plt.loglog(bins, count)
plt.xlabel('duration')
plt.ylabel('count')
plt.title('0:'+str(cut1))

plt.subplot(222)
num_bins = 60
n, bins, patches = plt.hist(list(
    [x for x in [x.duration for x in toy_list[int(cut1/sample_factor):int(cut2/sample_factor)]] if x < 10*60]),
    num_bins, facecolor='green')
#count = mlab.normpdf(bins, 100, 15)
#plt.loglog(bins, count)
plt.xlabel('duration')
plt.ylabel('count')
plt.title(str(cut1)+':'+str(cut2))

plt.subplot(223)
num_bins = 60
n, bins, patches = plt.hist(list(
    [x.duration for x in toy_list[int(cut2/sample_factor):int(cut3/sample_factor)]]),
    num_bins, facecolor='green')
#count = mlab.normpdf(bins, 100, 15)
#plt.loglog(bins, count)
plt.xlabel('duration')
plt.ylabel('count')
plt.title(str(cut2)+':'+str(cut3))

plt.subplot(224)
num_bins = 60
n, bins, patches = plt.hist(list(
    [x for x in [x.duration for x in toy_list[int(cut3/sample_factor):]] if x < 10*60]),
    num_bins, facecolor='green')
#count = mlab.normpdf(bins, 100, 15)
#plt.loglog(bins, count)
plt.xlabel('duration')
plt.ylabel('count')
plt.title(str(cut3)+':10000000')

plt.show()
