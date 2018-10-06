import matplotlib.pyplot as plt

f = open("train_small.csv", 'r')
g = open("test_small.csv", 'r')

#Skip the head line
f.readline()
g.readline()

################################################

train_label_data, train_feature_data = [], []

for line in f:
	labels = line.split(',')
	labels = list(map(str.strip, labels))
	feature = labels[-1].split(' ')
	labels[-1] = feature[0]
	feature = feature[1:]
	labels = list(map(int, labels))
	train_label_data.append(labels)
	feature = [list(map(int, x.split(':'))) for x in feature]
	feature = dict(feature)
	train_feature_data.append(feature)

f.close()

#################################################

test_feature_data = []

for line in g:
	feature = line.split(' ')
	feature = feature[1:]
	feature = [list(map(int, x.split(':'))) for x in feature]
	feature = dict(feature)
	test_feature_data.append(feature)

g.close()

#################################################

train_value_merge = [1700]
for feature in train_feature_data:
	train_value_merge.extend(list(feature.values()))

test_value_merge = [1700]
for feature in test_feature_data:
	test_value_merge.extend(list(feature.values()))

train_value_merge = [x for x in train_value_merge if x>1 and x<22]
test_value_merge = [x for x in test_value_merge if x>1 and x<22]

plt.hist(train_value_merge, 20, facecolor='r', alpha=0.75)
plt.hist(test_value_merge, 20, facecolor='g', alpha=0.75)
plt.show()
