from hierarchy_common import *
from hierarchy_functions import all_leaves
from copy import deepcopy

f = open("../data/hierarchy.txt", 'r')

descendant_tree = {}
ancestor_tree = {}

for line in f:
	(parent, child) = list(map(int, line.split(' ')))
	try: descendant_tree[parent].add(child)
	except: descendant_tree[parent] = set([child])
	try: ancestor_tree[child].add(parent)
	except: ancestor_tree[child] = set([parent])

f.close()

leaves = set(ancestor_tree.keys()).difference(list(descendant_tree.keys()))
intermediate = set(descendant_tree.keys())
roots = intermediate.difference(list(ancestor_tree.keys()))

"""
print "Statistics before removing:"

print "Number of leaves:", len(leaves)
print "Number of intermediate nodes:", len(intermediate)
print "Number of roots:", len(roots)
"""

### add those roots, because otherwise some leaves will be missing ---------------------------------------
roots.add(1007112) # this node go and backward w/ 1107113 and 1001195
roots.add(2145288) # this node go and backward w/ 2303351
roots.add(2062360) # ...
roots.add(1042969)
roots.add(2111381)
roots.add(2370074)
roots.add(2103289)
roots.add(2073638)
roots.add(2284099)
roots.add(2085216)
roots.add(1102164)
roots.add(2167982)
roots.add(2341554)
roots.add(2156741)
roots.add(1010926)
roots.add(2131206)
roots.add(2009938)
roots.add(1007481)
roots.add(2070885)
roots.add(2093738)

### break some big roots ---------------------------------------------------------------------------------
### inputs come from hierarchy_functions.py: leaf_distribution_for_roots.png

### handle root 1001505, because it has 129062 leaves.
roots.update(set(filter(is_not_leaf, descendant_tree[1001505])))
#roots.add(1001599)
#roots.add(1002400)
#roots.add(2083890)

### handle node 1001599 (child of 1001505), because it has 77900 leaves.
roots.update(set(filter(is_not_leaf, descendant_tree[1001599])))

### handle node 1000143 (child of 1001599)
roots.update(set(filter(is_not_leaf, descendant_tree[1000143])))
#roots.add(2004048)

### merge trees ------------------------------------------------------------------------------------------
### inputs come from hierarchy_functions.py: check "all_leaves" for some extremely popular label groups
"""
def merge_trees(new_root, merge_root_lst):
	roots.add(new_root)
	descendant_tree[new_root] = set(merge_root_lst)
	roots.difference_update(set(merge_root_lst))

### for the most popular leaves 24177 (support 4.12%) & 285613
merge_trees(4000001, [2287120, 2214145, 2322170, 2023611, 2299629, 2178462, 2394370, 2300073, 2026257, 2325141, 2150327, 2441305, 2198074, 2341276])
### for several popular leaves
#merge_trees(4000002, [1001441, 1004020])

### from the information of "popular_rootgroups.sh/txt"
merge_trees(5000001, [1000117, 1000116, 1000115, 1000042, 1000114])
merge_trees(5000002, [2080279, 2280282, 2389498, 2032837, 1007636, 2332707, 1002359, 2278809])
"""

### delete loops (actually make trees without crossing) from the root ------------------------------------

def hierarchy_to_trees (roots, descendant_tree, passed_nodes, exact_tree=True):
	last_step_nodes = deepcopy(roots)
	for i in range(20):
		next_step_nodes = set()
		print(len(last_step_nodes))
		for node in last_step_nodes:
			try: descendants = descendant_tree[node].difference(passed_nodes)
			except: continue
			if exact_tree == True: descendants = descendants.difference(next_step_nodes)
			if not descendants: del descendant_tree[node]
			elif descendants != descendant_tree[node]: descendant_tree[node] = descendants
			next_step_nodes.update(descendants)
		passed_nodes.update(next_step_nodes)
		last_step_nodes = next_step_nodes

### first round: roughly find all the trees, divide them into big trees and small trees
passed_nodes = deepcopy(roots)
duplicate_tree = deepcopy(descendant_tree)
hierarchy_to_trees (roots, duplicate_tree, passed_nodes, exact_tree=True)

descendant_tree = duplicate_tree

### second round: organize trees from size large to small
### (but if we do that, some leaves will be missing when count in read_hierarchy.py
### I don't understand why, so just remove this trick)
"""
big_roots = set()
#median_roots = set()
#small_roots = set()
for root in roots:
	tree_leaves = all_leaves(root, duplicate_tree)
	big_roots.add(root)
	#if len(tree_leaves) >= 0: big_roots.add(root)
	#elif len(tree_leaves) > 50: median_roots.add(root)
	#else: small_roots.add(root)

passed_nodes = deepcopy(big_roots)
hierarchy_to_trees (big_roots, descendant_tree, passed_nodes, exact_tree=True)

passed_nodes.update(median_roots)
hierarchy_to_trees (median_roots, descendant_tree, passed_nodes, exact_tree=True)
passed_nodes.update(small_roots)
hierarchy_to_trees (small_roots, descendant_tree, passed_nodes, exact_tree=True)
"""

useless = intermediate.difference(passed_nodes)
for node in useless: del descendant_tree[node]

### merge trees: -----------------------------------------------------------------------------------------
### code comes from popular_rootgroups.sh > popular_rootgroup_handler.py > merge_rootgroups.txt

"""
descendant_tree[3000000] = [1031388, 2341487]
descendant_tree[3000001] = [1007138, 1091996]
descendant_tree[3000002] = [1018031, 1006103]
descendant_tree[3000003] = [2397994, 2080279]
descendant_tree[3000004] = [1007576, 2046483]
descendant_tree[3000005] = [1009864, 1038312, 1000062]
descendant_tree[3000006] = [2218751, 2098607, 2073071]
descendant_tree[3000007] = [1006673, 2414389]
"""

### write ------------------------------------------------------------------------------------------------
g = open("data/forest.txt", 'w')

for parent, children in descendant_tree.items():
	for child in children: g.write(str(parent)+' '+str(child)+'\n')

g.close()

