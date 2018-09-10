f = open("feature_counts.txt", 'r')

data = []

for line in f:
	feature, count = line.split(' ')
	data.append([int(feature), int(count)])

f.close()

length = 2365436

data.sort(key=lambda x: x[1], reverse=True)

g = open("popular_features.txt", 'w')

for x in data[:500]:
	g.write(str(x[0])+' '+str(1.0*x[1]/length)+'\n')

g.close()
