from itertools import izip

f = open("pred_statistics_knn.csv", 'r')
#g = open("pred_knn.csv", 'r')
g = open("pred.csv", 'r')

count = 0
MaP = 0
MaR = 0

for lineF, lineG in izip(f, g):
	item_idF, labelF = lineF.strip().split(',')
	item_idG, labelG = lineG.strip().split(',')
	labelF = [x.split(':') for x in labelF.split(' ')]
	labelF = set(zip(*labelF)[0])
	labelG = set(labelG.split(' '))

	tp = len(labelF.intersection(labelG))
	fn = len(labelF.difference(labelG))
	fp = len(labelG.difference(labelF))

	if tp != 0:
		MaP += 1.0*tp/(tp+fp)
		MaR += 1.0*tp/(tp+fn)

	count += 1

MaP = MaP/count
MaR = MaR/count

print 2*MaP*MaR/(MaP+MaR)

f.close()
g.close()
