f = open("pred_votevalue_knn.csv", 'r')
g = open("pred_knn.csv", 'w')

f.readline()
g.write('Id,Predicted\n')

for line in f:
	item_id, label = line.strip().split(',')
	label = [x.split(':') for x in label.split(' ')]
	pred = []
	## (i)
	#pred = zip(*label)[0][:3]
	## (ii)
	for i in range(len(label)):
		label[i][1] = float(label[i][1])
		if label[0][1] == 0:
			pred.append("24177")
			break
		if i == 0 or (i == 1 and label[i][1]/label[0][1] > 0.4) or (i == 2 and label[i][1]/label[0][1] > 0.55) or (i == 3 and label[i][1]/label[0][1] > 0.60) or (i >= 4 and label[i][1]/label[0][1] > 0.7):
			pred.append(label[i][0])
	g.write(item_id+','+' '.join(pred)+'\n')

f.close()
g.close()

# SCORES:
# (i) => 0.231582279541
# (i+rule) => 0.243007673654
# (ii) => 0.263585100543
# (ii+rule) => 0.27673443464
