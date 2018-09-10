N_OF_TAGS = 5

popular_labels = [24177, 285613, 98808, 264962, 167593]

popular_labels_count = dict(zip(popular_labels, [0]*10))
count = 0

f = open("pred_statistics_knn.csv", 'r')

#Skip the head line
f.readline()

for line in f:
    tmp = line.split(',')
    tmp = tmp[1].split(' ')
    tmp.pop()
    for item in tmp:
        label, rank = item.split(':')
        if 0 < int(rank) < N_OF_TAGS:
            try:
                popular_labels_count[int(label)] += 1
            except:
                pass
    count += 1

f.close()

for label in popular_labels_count:
    print label, 1.0*popular_labels_count[label]/count

#98808 0.00338123415046
#24177 0.12045646661
#264962 0.00253592561285
#285613 0.00211327134404
#167593 0.00126796280642
