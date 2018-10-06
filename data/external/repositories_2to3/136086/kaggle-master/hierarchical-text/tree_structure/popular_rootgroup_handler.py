from hierarchy_functions import *
from read_hierarchy import *
from operator import itemgetter

### remove root groups with so big roots -----------------------------------------------------------------

leaves_count = {}
for root in roots:
	tree_leaves = all_leaves(root, descendant_tree)
	if len(tree_leaves) > 500: leaves_count[root] = len(tree_leaves)

leaves_count = sorted(iter(leaves_count.items()), key=itemgetter(1), reverse=True)
#print "\n\nMost popular roots (w/ leaf #):", leaves_count[:10]

most_popular_roots = set(zip(*leaves_count[:20])[0])
popular_roots = set(zip(*leaves_count)[0])

#most_popular_roots.add(4000001) # the root for the two most popular labels (so large support)

f = open("popular_rootgroups_tmp.txt", 'r')
g = open("popular_rootgroups.txt", 'w')

for line in f:
	roots = line.split(' ')
	roots.pop() # delete the last item, i.e., popularity information
	roots = list(map(int, roots))
	if len(set(roots).intersection(most_popular_roots)) == 0 and len(set(roots).intersection(popular_roots)) < 2: 
		g.write(line)

f.close()
g.close()

### combine root groups to larger combinations -----------------------------------------------------------

f = open("popular_rootgroups.txt", 'r')

root_groups = []
for line in f:
	roots = line.split(' ')
	roots.pop()
	roots = set(map(int, roots))
	root_groups.append(roots)

f.close()

merge_groups = []
for i in range(len(root_groups)):
	roots = root_groups[i]
	merge = False
	for j in range(i+1, len(root_groups)):
		if root_groups[j].intersection(roots):
			root_groups[j].update(roots)
			merge = True
	if merge == False: merge_groups.append(roots)

g = open("merge_rootgroups.txt", 'w')

for i in range(len(merge_groups)):
	roots = merge_groups[i]
	g.write("descendant_tree["+str(3000000+i)+"] = [" + ', '.join(map(str, list(roots))) + "]\n")

g.close()
