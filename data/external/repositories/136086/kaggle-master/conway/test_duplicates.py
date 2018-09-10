import csv

trainDic = {}

with open('train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader, None)
	for row in reader:
		train_start = int(''.join(row[2:402]), 2)
		train_end = int(''.join(row[402:]), 2)
		trainDic[(train_end, row[1])] = train_start
		# If same "train_end" and "delta", just overwrite with the last "train_start".
		# The difference between the two sets are tiny, maybe rised by all blank cases.
		# But there seems no blank cases in both train and test set.

print(len(trainDic)) 
# train set has 47798 unique elements 
# (if count "train_end" and "delta", then 47044 unique elements)
# (if only count "train_end", then 46024 unique elements)

count = 0
with open('test.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader, None)
	for row in reader:
		test = int(''.join(row[2:]), 2)
		if (test, row[1]) in trainDic:
			#print(trainDic[test], int(row[1]), test)
			#print("find duplication!!", test)
			count += 1

print(count) 
# count "train_end" and "delta", there are 3641 duplicates.
# (if don't count "delta", then 4xxx; so "delta" is also not random)

