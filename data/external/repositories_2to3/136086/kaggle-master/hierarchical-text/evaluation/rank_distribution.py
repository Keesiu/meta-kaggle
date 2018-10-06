import matplotlib.pyplot as plt
import numpy as np

#f = open("pred_statistics_knn.csv", 'r')
f = open("pred_statistics_largest.csv", 'r')

#Skip the head line
f.readline()

count = [0]*101
allcount = 0

for line in f:
	tmp = line.split(',')
	if tmp[1]:
		tmp = tmp[1].split(' ')
		tmp.pop()
		rank = [int(x.split(':')[1]) for x in tmp]
		allcount += 1
		for i in rank:
			count[i] += 1

count = [1.0*x/allcount for x in count]
plt.bar(np.arange(0.5,14.5,1), count[1:15], width=1)
plt.axis([0.5, 14.5, 0, 0.4])
plt.xlabel('rank of label')
plt.ylabel('pertage is scope')
plt.show()
    
