from hierarchy_common import *

if __name__ == '__main__':

	from read_hierarchy import *
	from read_train import *
	from math import log10
	from numpy import mean
	from operator import itemgetter
	import matplotlib.pyplot as plt

	"""
	print common_ancestor ([314523, 165538, 76255, 335416], ancestor_tree)
	print ancestor_path (335416, 2216162, ancestor_tree)
	"""

	### roots for most popular labels
	"""
	for node in [24177, 285613, 98808, 264962, 52954, 242532, 78249, 220514, 444502, 237290]:
		rt = all_roots(node, ancestor_tree)
		print rt, len(all_leaves(list(rt)[0], descendant_tree))

	#print all_roots(167593, ancestor_tree)
	#print all_roots(300558, ancestor_tree)
	#print all_roots(337728, ancestor_tree)
	#print all_roots(73518, ancestor_tree)
	#(10721, 79), (327590, 79), (24016, 72), (174545, 72), 
	#"""

	### check "all_leaves" for some extremely popular label groups
	"""
	### the most popular leave, 24177 has roots #set([2287120, 2214145, 2322170, 2023611, 2299629, 2178462]),
	### but all of them are small ones, support 4.12%, but all only have 39 leaves. So just merge them.
	for root in [2287120, 2214145, 2322170, 2023611, 2299629, 2178462]:
		tree_leaves = all_leaves(root, descendant_tree)
		print root, len(tree_leaves)
	for root in [3000001, 2394370, 2300073, 2026257, 2325141, 2150327, 2441305, 2198074, 2341276]:
		tree_leaves = all_leaves(root, descendant_tree)
		print root, len(tree_leaves)
	"""

	### some statistics for tree structure
	#"""
	leaves_count = {}
	for root in roots:
		tree_leaves = all_leaves(root, descendant_tree)
		if len(tree_leaves) > 0: leaves_count[root] = len(tree_leaves)

	leaves_count_lst = list(zip(*iter(leaves_count.items())))
	
	### leaf_distribution_for_roots.png
	#plt.hist(map(log10, leaves_count_lst[1]))
	#plt.title('leaf distribution for roots (after remove loops)')
	#plt.xlabel('log10(# of leaves)')
	#plt.ylabel('popularity of roots')
	#plt.show()

	leaves_count = sorted(iter(leaves_count.items()), key=itemgetter(1), reverse=True)
	print("\n\nMost popular roots (w/ leaf #):")
	for i in range(5): print(leaves_count[i])
	# After foresting, but before merging trees in forest
	#(1001039, 9854)
	#(1002416, 9204)
	#(1000144, 8494)
	#(2278809, 7532)
	#(1004285, 7369)
	# After merging (threshold 0.02, 4093 rules)
	#(3000170, 19998)
	#(3000162, 19982)
	#(3000139, 19978)
	#(3000149, 19932)
	#(3000172, 19919)

	# Merging only from the label
	#(4002468, 3633)
	#(4002250, 255)
	#(4001964, 77)
	#(4001733, 49)
	#(4002262, 48)

	#"""

	### distance_to_common_ancestor.png 
	### (speed goes up significantly after we removed the loops in hierarchy)
	#"""
	common_ancestor_distance = [[], [], [], [], [], []]
	occupation_table = [[0]*6, [0]*6]
	for labels in label_data:
		if 2 <= len(labels) <= 5:
			occupation_table[0][len(labels)] += 1
			ancestors = common_ancestor(labels, ancestor_tree)
			if ancestors:
				path_length = min([mean([len(ancestor_path(label, ancestor, ancestor_tree)) for label in labels]) for ancestor in ancestors])
				#print len(labels), path_length
				common_ancestor_distance[len(labels)].append(path_length)
				occupation_table[1][len(labels)] += 1

	print("\n\nprecentage of entries which have common ancestors:")
	print("2 labels:", 1.0*occupation_table[1][2]/occupation_table[0][2])
	print("3 labels:", 1.0*occupation_table[1][3]/occupation_table[0][3])
	print("4 labels:", 1.0*occupation_table[1][4]/occupation_table[0][4])
	print("5 labels:", 1.0*occupation_table[1][5]/occupation_table[0][5])

	# precentage of entries which have common ancestors:
	# Before removing loops
	#2 labels: 0.841827768014
	#3 labels: 0.765432098765
	#4 labels: 0.71484375
	#5 labels: 0.646408839779
	# After foresting, but before merging trees in forest
	#2 labels: 0.192133405837
	#3 labels: 0.0456411494808
	#4 labels: 0.0165084605861
	#5 labels: 0.0059880239521
	# After merging (threshold 0.02, 4093 rules)
	#2 labels: 0.388979517854
	#3 labels: 0.217338807051
	#4 labels: 0.104828724721
	#5 labels: 0.0724550898204

	# Merging only from the label
	#2 labels: 0.094797897408
	#3 labels: 0.0601304032842
	#4 labels: 0.0206355757326
	#5 labels: 0.0155688622754

	plt.subplot(221)
	plt.hist(common_ancestor_distance[2], bins=12, range=(0, 12))
	plt.title('2 labels')
	plt.xlabel('distance to common ancestor')
	plt.ylabel('# of entries')

	plt.subplot(222)
	plt.hist(common_ancestor_distance[3], bins=12, range=(0, 12))
	plt.title('3 labels')
	plt.xlabel('distance to common ancestor')
	plt.ylabel('# of entries')

	plt.subplot(223)
	plt.hist(common_ancestor_distance[4], bins=12, range=(0, 12))
	plt.title('4 labels')
	plt.xlabel('distance to common ancestor')
	plt.ylabel('# of entries')

	plt.subplot(224)
	plt.hist(common_ancestor_distance[5], bins=12, range=(0, 12))
	plt.title('5 labels')
	plt.xlabel('distance to common ancestor')
	plt.ylabel('# of entries')

	plt.show()
	#"""
