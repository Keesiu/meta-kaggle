f = open("data/merged_forest.txt", 'r')
#f = open("data/forest.txt", 'r')
#f = open("data/hierarchy.txt", 'r')
#f = open("data/merged_labeltrees.txt", 'r')

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

leaves = set(ancestor_tree.keys()).difference(list(descendant_tree.keys()))
intermediate = set(descendant_tree.keys())
roots = intermediate.difference(list(ancestor_tree.keys()))

### roots who's children are not all leaves (4866 in all)
roots_nontrivial = set()
for root in roots:
	if len([x for x in descendant_tree[root] if x > 1000000]) > 0: roots_nontrivial.add(root)

leaves = [leave for leave in leaves if leave < 1000000]

### ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	#print(descendant_tree)
	#print(ancestor_tree)

	#print roots
	#print roots_nontrivial

	print("Number of leaves:", len(leaves))
	print("Number of intermediate nodes:", len(intermediate))
	print("Number of roots:", len(roots))

	pass
