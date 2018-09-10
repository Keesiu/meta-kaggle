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

train_value_merge = [1700]
for feature in train_feature_data:
	train_value_merge.extend(feature.values())

test_value_merge = [1700]
for feature in test_feature_data:
	test_value_merge.extend(feature.values())

train_value_merge = filter(lambda x: x>1 and x<22, train_value_merge)
test_value_merge = filter(lambda x: x>1 and x<22, test_value_merge)

plt.hist(train_value_merge, 20, facecolor='r', alpha=0.75)
plt.hist(test_value_merge, 20, facecolor='g', alpha=0.75)
plt.show()
