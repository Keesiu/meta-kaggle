import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from load_toys_rev import *

filename = 'data/toys_rev2_sample10.csv'
sample_factor = 10
toy_list = load_toys_rev(filename)

timesum_cut = [150,600,2400,22500]
timesum = [0, 0, 0, 0]

for toy in toy_list:
    for i in range(len(timesum_cut)):
        if toy.duration < timesum_cut[i]:
            timesum[i] += toy.duration*sample_factor

print(timesum)
# [269583430, 604000370, 1062934090, 26010683440]

print((list([x/26010683440 for x in timesum])))
# [0.010364334740448481, 0.023221241817550642, 0.04086528877458823, 1.0]
