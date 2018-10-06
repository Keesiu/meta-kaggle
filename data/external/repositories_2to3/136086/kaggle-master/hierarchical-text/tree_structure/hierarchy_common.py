def is_leaf(node):
	if node < 1000000: return True
	else: return False

def is_not_leaf(node):
	if node < 1000000: return False
	else: return True

def ancestor_path(descendant, ancestor, ancestor_tree):
	def ancestor_path_recur (paths, ancestor):
		paths_nextstep = []
		for path in paths:
			try: 
				ancestors = ancestor_tree[path[-1]]
				if ancestor in ancestors:  return path+[ancestor]
				else: paths_nextstep.extend([path+[x] for x in ancestors])
			except: pass
		return ancestor_path_recur(paths_nextstep, ancestor)
	if descendant == ancestor: return [descendant]
	else: return ancestor_path_recur([[descendant]], ancestor)

### this way works in case without loop.
def all_roots(node, ancestor_tree):
	try: parents = ancestor_tree[node]
	except: return set([node])
	root_lst = [all_roots(parent, ancestor_tree) for parent in parents]
	return set.union(*root_lst)

def common_ancestor(descendant_lst, ancestor_tree):
	### can only find out common ancestor which is at most 8 steps away
	ancestors_lst = [set([x]) for x in descendant_lst]
	depth = 1
	while not set.intersection(*ancestors_lst) and depth <= 12:
		tmp = ancestors_lst
		for i in range(len(ancestors_lst)):
			for x in ancestors_lst[i]: 
				try: tmp[i] = tmp[i].union(ancestor_tree[x])
				except: pass
		ancestors_lst = tmp
		depth += 1
	return set.intersection(*ancestors_lst)

### this way works in case without loop. extremely quickly.
#"""
def all_leaves(node, descendant_tree):
	if is_leaf(node): return set([node])

	try: children = descendant_tree[node]
	except: return set()

	leaf_lst = [all_leaves(child, descendant_tree) for child in children]
	if leaf_lst: return set.union(*leaf_lst)
	else: return set()
#"""

### this way goes so slowly in case the hierarchy has complicated structure (a.k.a. before removing loops)
"""
def all_leaves(node, descendant_tree):
	def all_leaves_recur(node, mids):
		if is_leaf(node): return set([node])

		try: descendants = descendant_tree[node]
		except: return set()

		descendants.difference_update(mids)
		mids.update(descendants)
		leaf_lst = map(lambda descendant: all_leaves_recur(descendant, mids), descendants)
		#print descendants
		if leaf_lst: return set.union(*leaf_lst)
		else: return set()
	return all_leaves_recur(node, set())
#"""

### this way goes so slowly in case the hierarchy is simple (a.k.a. after removing loops)
"""
def all_leaves(node, descendant_tree):
	node_array = [0]*2445721
	def all_leaves_recur(node):
		if is_leaf(node): node_array[node] = 1
		elif node_array[node] == 2: pass
		else:
			node_array[node] = 2
			try: descendants = descendant_tree[node]
			except: return
			for descendant in descendants: all_leaves_recur(descendant)
	all_leaves_recur(node)
	return [i for i,x in enumerate(node_array) if x == 1]
#"""
