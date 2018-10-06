from operator import itemgetter
from numpy import prod

from constants import TRAIN_SIZE
from hierarchy_common import all_roots, all_leaves

descendant_tree = {}
ancestor_tree = {}

### ------------------------------------------------------------------------------------------------------

f = open("data/label_popularity.txt", 'r')

label_popularity = {}
for line in f:
	(label, count) = line.split(' ')
	label_popularity[int(label)] = int(count)

f.close()

### ------------------------------------------------------------------------------------------------------

f = open("data/popular_labelgroups.txt", 'r')

label_groups = []
for line in f:
	labels = line.split(' ')
	support = labels.pop()
	support = float(support[1:-2])
	labels = set(map(int, labels))
	significance = (TRAIN_SIZE*support - 1)/prod([label_popularity[node] for node in labels])
	# "-1" makes smaller support less significant
	label_groups.append((labels, significance))
	#label_groups.append((labels, support))

f.close()

### ------------------------------------------------------------------------------------------------------

label_groups.sort(key=itemgetter(1), reverse=True)

print((label_groups[:10]))

i = 1
for (roots, support) in label_groups[:50000]:
	real_roots = set.union(*[all_roots(node, ancestor_tree) for node in roots])
	if len(real_roots) == 1: continue
	leaves_count = sum([len(all_leaves(node, descendant_tree)) for node in real_roots])
	if leaves_count < 20000: 
		added_roots = [x for x in real_roots if x > 4000000]
		original_roots = [x for x in real_roots if x < 4000000]
		if len(added_roots) == 1:
			descendant_tree[added_roots[0]].update(set(original_roots))
			for node in original_roots: ancestor_tree[node] = set([added_roots[0]])
		else:
			descendant_tree[4000000+i] = real_roots
			for node in real_roots: ancestor_tree[node] = set([4000000+i])
			i += 1
	#else: print "can't merge", roots, real_roots


### write ------------------------------------------------------------------------------------------------
g = open("data/merged_labeltrees.txt", 'w')

for parent, children in descendant_tree.items():
	for child in children: g.write(str(parent)+' '+str(child)+'\n')

g.close()
