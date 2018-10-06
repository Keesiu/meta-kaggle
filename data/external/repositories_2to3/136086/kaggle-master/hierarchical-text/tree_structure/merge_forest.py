from operator import itemgetter
from numpy import prod

from constants import TRAIN_SIZE
from hierarchy_common import all_roots, all_leaves

f = open("data/forest.txt", 'r')

descendant_tree = {}
ancestor_tree = {}

for line in f:
	(parent, child) = list(map(int, line.split(' ')))
	try: descendant_tree[parent].add(child)
	except: descendant_tree[parent] = set([child])
	try: ancestor_tree[child].add(parent)
	except: ancestor_tree[child] = set([parent])

f.close()

### ------------------------------------------------------------------------------------------------------

f = open("data/label_popularity.txt", 'r')

root_popularity = {}
for line in f:
	(label, count) = line.split(' ')
	node = all_roots(int(label), ancestor_tree).pop()
	try: root_popularity[node] += int(count)
	except: root_popularity[node] = int(count)

f.close()

#root_popularity = sorted(root_popularity.iteritems(), key=itemgetter(1), reverse=True)
#print(root_popularity[:10])

### ------------------------------------------------------------------------------------------------------

f = open("data/popular_rootgroups.txt", 'r')

root_groups = []
for line in f:
	roots = line.split(' ')
	support = roots.pop()
	support = float(support[1:-2])
	roots = set(map(int, roots))
	significance = (TRAIN_SIZE*support - 1)/prod([root_popularity[node] for node in roots])
	# "-1" makes smaller support less significant
	root_groups.append((roots, significance))
	#root_groups.append((roots, support))

f.close()

### ------------------------------------------------------------------------------------------------------

root_groups.sort(key=itemgetter(1), reverse=True)
#print(root_groups[:10])

i = 1
for (roots, support) in root_groups:
	real_roots = set.union(*[all_roots(node, ancestor_tree) for node in roots])
	if len(real_roots) == 1: continue
	leaves_count = sum([len(all_leaves(node, descendant_tree)) for node in real_roots])
	if leaves_count < 20000: 
		added_roots = [x for x in real_roots if x > 3000000]
		original_roots = [x for x in real_roots if x < 3000000]
		if len(added_roots) == 1:
			descendant_tree[added_roots[0]].update(set(original_roots))
			for node in original_roots: ancestor_tree[node] = set([added_roots[0]])
		else:
			descendant_tree[3000000+i] = real_roots
			for node in real_roots: ancestor_tree[node] = set([3000000+i])
			i += 1
	#else: print "can't merge", roots, real_roots

### write ------------------------------------------------------------------------------------------------

g = open("data/merged_forest.txt", 'w')

for parent, children in descendant_tree.items():
	for child in children: g.write(str(parent)+' '+str(child)+'\n')

g.close()

