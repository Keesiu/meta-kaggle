f = open("../data/train_1-100.csv", 'r')

#Skip the head line
f.readline()

label_data, feature_data = [], []

for line in f:
	labels = line.split(',')
	labels = list(map(str.strip, labels))
	feature = labels[-1].split(' ')
	labels[-1] = feature[0]
	feature = feature[1:]
	labels = list(map(int, labels))
	label_data.append(labels)
	feature = [list(map(int, x.split(':'))) for x in feature]
	feature = dict(feature)
	feature_data.append(feature)

f.close()

train_size = len(label_data)

label_popularity = {}
for labels in label_data:
	for label in labels:
		try: label_popularity[label] += 1
		except: label_popularity[label] = 1

g = open("data/label_popularity.txt", 'w')
for label, count in label_popularity.items(): g.write(str(label)+' '+str(count)+'\n')
g.close()

feature_popularity = {}
for features in feature_data:
	for feature in features:
		try: feature_popularity[feature] += 1
		except: feature_popularity[feature] = 1

g = open("data/feature_popularity.txt", 'w')
for feature, count in feature_popularity.items(): g.write(str(feature)+' '+str(count)+'\n')
g.close()

if __name__ == '__main__':

	#print(label_data)
	#print(feature_data)
	#print(map(len, label_data))

	import matplotlib.pyplot as plt
	from operator import itemgetter

	### label_length_distribution.png
	"""
	plt.hist(map(len, label_data), bins=30, range=(0.5, 30.5))
	plt.title('distribution of # of labels for training entries')
	plt.xlabel('# of labels')
	plt.ylabel('# of entries')
	plt.show()
	#"""

	### feature_length_distribution.png
	"""
	plt.hist(map(len, feature_data), bins=25, range=(0, 300))
	plt.title('distribution of # of features for training entries')
	plt.xlabel('# of features')
	plt.ylabel('# of entries')
	plt.show()
	#"""

	# NOTE!!!!!!!!!!
	# THE DATA SET IS NOT UNIFORM AT ALL

	sorted_labels = sorted(iter(label_popularity.items()), key=itemgetter(1), reverse=True)
	print((sorted_labels[:20]))
	#--first 20000--
	#[(24177, 874), (389005, 168), (264962, 167), (413645, 140), (438906, 115), 
	#(150101, 114), (391405, 114), (9089, 100), (242532, 95), (1171, 92), (305109, 92), 
	#(195420, 88), (137379, 85), (98808, 85), (396560, 84), (139391, 80), (333275, 76), 
	#(352578, 75), (175604, 72), (362193, 72)]
	#--last 20000--
	#[(24177, 3862), (174425, 728), (419276, 486), (174595, 474), (383600, 272), 
	#(126336, 257), (126782, 193), (98808, 170), (8840, 169), (10721, 147), (285613, 146), 
	#(445621, 141), (87256, 136), (263233, 126), (300558, 119), (167593, 111), 
	#(246293, 107), (214145, 104), (59758, 100), (78249, 99)]
	#--23655 (x100)--
	#[(24177, 3894), (285613, 384), (98808, 136), (264962, 123), (52954, 108), (242532, 102), 
	#(78249, 99), (220514, 95), (444502, 91), (237290, 88), (167593, 87), (300558, 83), 
	#(337728, 82), (73518, 80), (10721, 79), (327590, 79), (24016, 72), (174545, 72), 
	#(87241, 55), (374771, 55)]

	sorted_features = sorted(iter(feature_popularity.items()), key=itemgetter(1), reverse=True)
	print((sorted_features[:20]))
	#--first 20000--
	#[(1657857, 4555), (1662756, 4492), (1055696, 3844), (1647082, 3694), (245929, 3438), 
	#(1609223, 3413), (517668, 3315), (760008, 3314), (1822977, 3169), (444833, 3006), 
	#(77076, 2996), (1357328, 2983), (572336, 2968), (1972926, 2836), (2015694, 2827), 
	#(1550210, 2823), (1257203, 2791), (660115, 2786), (2021871, 2557), (1673413, 2547)]
	#--last 20000--
	#[(634374, 3431), (1257203, 2561), (173875, 2289), (1055696, 2274), (1662756, 2251), 
	#(1832029, 2075), (760008, 2054), (1661467, 2013), (1247614, 1977), (1822977, 1774), 
	#(1794479, 1771), (343843, 1755), (1657857, 1737), (2021871, 1734), (1884028, 1627), 
	#(245929, 1606), (1972926, 1594), (1054595, 1593), (444833, 1549), (1647082, 1515)]
	#--23655 random(x100)--
	#[(634374, 3653), (1257203, 3585), (1055696, 3397), (1662756, 2909), (173875, 2902), 
	#(760008, 2865), (1657857, 2819), (1647082, 2714), (1822977, 2536), (343843, 2441), 
	#(1794479, 2422), (1832029, 2367), (245929, 2291), (2021871, 2270), (1661467, 2263), 
	#(1972926, 2232), (343448, 2061), (1676819, 2012), (331158, 2010), (823758, 2002)]
