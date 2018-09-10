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

f = open("../data/test.csv", 'r')

#Skip the head line
f.readline()

print "Id,Predicted"

for line in f:
	predict = set()
	feature = line.split(' ')
	item_id = feature[0].split(',')[0]
	feature = feature[1:]
	feature = map(lambda x: map(int, x.split(':')), feature)
	feature = set(zip(*feature)[0])
	#print feature

	for rule in rules:
		if rule[0].issubset(feature):
			predict.add(rule[1])

	print str(item_id)+','+' '.join(map(str, predict))

f.close()
