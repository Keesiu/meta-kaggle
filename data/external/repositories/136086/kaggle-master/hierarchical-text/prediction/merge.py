from itertools import izip

f = open('pred_knn.csv', 'r')
g = open('pred_rule.csv', 'r')

h = open('pred.csv', 'w')

f.readline()
g.readline()

h.write("Id,Predicted\n")

for lineF, lineG in izip(f, g):
	item_idF, labelF = lineF.strip().split(',')
	item_idG, labelG = lineG.strip().split(',')
	if item_idF != item_idG:
		assert 'error'
	labelF = set(map(int, labelF.split(' ')))
	if labelG != '':
		labelG = set(labelG.split(' '))
	else:
		labelG = set()
	#label = labelF.union(labelG)
	label = labelF.difference(set([24177, 98808, 264962, 167593])).union(labelG)
	if not label: label = set([24177])
	h.write(str(item_idF)+','+' '.join(map(str, label))+'\n')
