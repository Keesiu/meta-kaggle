import csv

trainDic = {}
with open('train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader, None)
	for row in reader:
		train_start = int(''.join(row[2:402]), 2)
		train_end = int(''.join(row[402:]), 2)
		trainDic[(train_end, row[1])] = train_start

testList = []
with open('test.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader, None)
	for row in reader:
		test = int(''.join(row[2:]), 2)
		testList.append((test, row[1]))

print((len(testList)))

f = open("submission.csv", 'r')
submissionList = f.read().split('\n')
f.close()

i = 0

g = open("submission_final.csv", 'w')
g.write(submissionList[0]+"\n")
for line in submissionList[1:50001]:
	idx = int(line.split(',')[0])
	if testList[idx-1][0] == 0: g.write(line)
	# but it seems that test set doesn't include all zero grid.
	elif testList[idx-1][0] == int(''.join(line.split(',')[2:]), 2): g.write(line)
	# if pattern, then keep it (this makes the duplicate number goes from 3641 to 265).
	elif testList[idx-1] in trainDic:
		pred = bin(trainDic[testList[idx-1]])[2:]
		if len(pred) == 400: g.write(str(idx) + ',' + ','.join(pred))
		else:
			zeros = ['0']*(400-len(pred))
			g.write(str(idx) + ',' + ','.join(zeros) + ',' + ','.join(pred))
		#print(testList[idx-1][0], trainDic[testList[idx-1]], int(''.join(line.split(',')[2:]), 2))
		i += 1
	else: g.write(line)
	g.write("\n")

g.close()

print(i)

# However, after count duplicates, the test score actually goes worse
# if not keep the pattern: 0.12477 -> 0.12488

# If keep the pattern, just improve a little bit: 0.12477 -> 0.12475
