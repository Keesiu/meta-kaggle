import csv

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def load_distribution(filename):
    distribution = [0]*25
    with open(filename, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #next(spamreader)
        for row in spamreader:
            if len(row) >= 2:
                distribution[int(int(row[0])/100)] += int(row[1])
                if (distribution[int(int(row[0])/100)] > 500000):
                    distribution[int(int(row[0])/100)] = 500000
                #distribution.append(int(row[1]))
    return distribution

plt.subplot(211)
usage_distribution = load_distribution("data/SmallToyUsageDistribution.txt")
plt.plot(usage_distribution,)
plt.xlabel('duration/100')
plt.ylabel('count')

plt.subplot(212)
toy_distribution = load_distribution("data/SmallToyDistribution.txt")
plt.plot(toy_distribution,)
plt.xlabel('duration/100')
plt.ylabel('count')

plt.show()
