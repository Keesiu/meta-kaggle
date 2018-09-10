from sys import argv
from subprocess import call, check_output

threshold1 = 0.2
threshold2 = 0.4

label = argv[1]
call(["sh", "popular_features.sh", label])

f = open("data/popular_features_"+label+'.csv', 'r')

rules = []

for line in f:
	features = line.split(' ')
	support = features.pop()
	tmp = check_output(["../rule/feature2labels", "../data/train_1-100.csv", str(threshold1)] + features)
	tmp = tmp.strip().split('\n')
	for output in tmp:
		words = output.split(' ')
		if (len(words) >= 7 and words[-3] == label):
			probablity = words[-1]
			rules.append([features, float(probablity)])

rules.sort(key=lambda x: x[1], reverse=True)

i = 0
for rule in rules:
	if (i > 10 and rule[1] < threshold2):
		break
	else:
		print ' '.join(rule[0]), "=>", label, "prob", rule[1]
		i += 1

f.close()
