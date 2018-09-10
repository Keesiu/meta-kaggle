import matplotlib.pyplot as plt

f = open("../data/train_small.csv", 'r')
g = open("../data/test_small.csv", 'r')

#Skip the head line
f.readline()
g.readline()

################################################

train_label_data, train_feature_data = [], []

for line in f:
	labels = line.split(',')
	labels = map(str.strip, labels)
	feature = labels[-1].split(' ')
	labels[-1] = feature[0]
	feature = feature[1:]
	labels = map(int, labels)
	train_label_data.append(labels)
	feature = map(lambda x: map(int, x.split(':')), feature)
	feature = dict(feature)
	train_feature_data.append(feature)

f.close()

#################################################

test_feature_data = []

for line in g:
	feature = line.split(' ')
	feature = feature[1:]
	feature = map(lambda x: map(int, x.split(':')), feature)
	feature = dict(feature)
	test_feature_data.append(feature)

g.close()

#################################################

train_feature_merge = []
for feature in train_feature_data:
	train_feature_merge.extend(feature.keys())

test_feature_merge = []
for feature in test_feature_data:
	test_feature_merge.extend(feature.keys())

plt.hist(train_feature_merge, 50, facecolor='r', alpha=0.75)
plt.hist(test_feature_merge, 50, facecolor='g', alpha=0.75)
plt.show()
