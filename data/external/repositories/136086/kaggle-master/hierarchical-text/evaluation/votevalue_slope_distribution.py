from itertools import izip
import numpy as np
import matplotlib.pyplot as plt

TEST_RANK = 1

f = open("pred_votevalue_knn.csv", 'r')
g = open("pred_statistics_knn.csv", 'r')

f.readline()
g.readline()

true_set = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]
false_set = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1]] # Elements "0" and "1" for scaling reason

for lineF, lineG in izip(f, g):
	item_idF, labelF = lineF.strip().split(',')
	item_idG, labelG = lineG.strip().split(',')
	labelF = [x.split(':') for x in labelF.split(' ')]
	labelG = dict([x.split(':') for x in labelG.split(' ')])

	for i in range(1,5):
		if labelF[i][0] in labelG.keys():
			true_set[i].append((float(labelF[i][1])+0.00001)/(float(labelF[i-1][1])+0.00001))
		elif labelF[i-1][0] in labelG.keys():
			false_set[i].append((float(labelF[i][1])+0.00001)/(float(labelF[i-1][1])+0.00001))

	#print labelF
	#print labelG

f.close()
g.close()

#print true_set
#print false_set

plt.figure(1)

plt.subplot(221)
plt.hist(false_set[1], 20, facecolor='r', alpha=0.75)
plt.hist(true_set[1], 20, facecolor='g', alpha=0.75)
plt.title('rank = 2')

plt.subplot(222)
plt.hist(false_set[2], 20, facecolor='r', alpha=0.75)
plt.hist(true_set[2], 20, facecolor='g', alpha=0.75)
plt.title('rank = 3')

plt.subplot(223)
plt.hist(false_set[3], 20, facecolor='r', alpha=0.75)
plt.hist(true_set[3], 20, facecolor='g', alpha=0.75)
plt.title('rank = 4')

plt.subplot(224)
plt.hist(false_set[4], 20, facecolor='r', alpha=0.75)
plt.hist(true_set[4], 20, facecolor='g', alpha=0.75)
plt.title('rank = 5')

plt.show()
