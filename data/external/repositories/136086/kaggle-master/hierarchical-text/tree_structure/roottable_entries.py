from read_hierarchy import *
from hierarchy_common import all_roots

f = open("data/train_small.csv", 'r')
g = open("data/roottable_entries.txt", 'w')

#Skip the head line
f.readline()

for line in f:
	labels = line.split(',')
	labels = map(str.strip, labels)
	feature = labels[-1].split(' ')
	labels[-1] = feature[0]
	labels = map(int, labels)
	if len(labels) <= 10:
		roots = set.union(*map(lambda label: all_roots(label, ancestor_tree), labels))
		g.write(' '.join(map(str, list(roots)))+'\n')

f.close()
