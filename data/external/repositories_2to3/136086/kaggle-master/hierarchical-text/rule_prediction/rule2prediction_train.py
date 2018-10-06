from sys import argv

rules = []

g = open("rule.txt", 'r')

for line in g:
	feature = line.split(' ')
	if feature[0] != '#':
		label = int(feature[-3])
		feature = set(map(int, feature[:-4]))
		rules.append((feature,label))

#print rules

g.close()

#################################################################

MaP = []
MaR = []

tp = 0
fn = 0
fp = 0

f = open("../data/train_1-100.csv", 'r')

#Skip the head line
f.readline()

nrow = 0
for line in f:
	predict = set()
	labels = line.split(',')
	labels = list(map(str.strip, labels))
	feature = labels[-1].split(' ')
	labels[-1] = feature[0]
	feature = feature[1:]
	labels = set(map(int, labels))
	feature = [list(map(int, x.split(':'))) for x in feature]
	feature = set(zip(*feature)[0])
	#print labels

	for rule in rules:
		if rule[0].issubset(feature):
			predict.add(rule[1])

	#labels = labels.intersection(set([24177]))
	labels = labels.intersection(set([98808]))
	#labels = labels.intersection(set([264962]))
	#labels = labels.intersection(set([167593]))

	#predict = predict.intersection(set([24177]))
	predict = predict.intersection(set([98808]))
	#predict = predict.intersection(set([264962]))
	#predict = predict.intersection(set([167593]))

	#tp = len(labels.intersection(predict))
	#fn = len(predict.difference(labels))
	#fp = len(labels.difference(predict))
	#print tp, fn, fp

	tp += len(labels.intersection(predict))
	fn += len(predict.difference(labels))
	fp += len(labels.difference(predict))

	"""
	if tp == 0: 
		MaP = 0
		MaR = 0
	else:
		MaP = 1.0*tp/(tp+fn)
		MaR = 1.0*tp/(tp+fp)
	print MaP, MaR
	"""

	#print predict
	nrow += 1
	#if nrow > 100: break

f.close()

print(tp, fn, fp)

MaP = 1.0*tp/(tp+fn)
MaR = 1.0*tp/(tp+fp)

print(2*MaP*MaR/(MaP+MaR))


