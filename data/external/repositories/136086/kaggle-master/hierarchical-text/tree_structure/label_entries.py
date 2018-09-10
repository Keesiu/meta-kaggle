f = open("../data/train_1-100.csv", 'r')
g = open("data/label_entries.txt", 'w')

#Skip the head line
f.readline()

for line in f:
	labels = line.split(',')
	labels = map(str.strip, labels)
	feature = labels[-1].split(' ')
	labels[-1] = feature[0]
	g.write(' '.join(labels)+'\n')
	labels = map(int, labels)

f.close()
