N_OF_TAGS = 5

popular_labels = [24177, 285613, 98808, 264962, 167593, 242532, 52954, 300558, 444502, 78249]

popular_labels_count = dict(list(zip(popular_labels, [0]*10)))
count = 0

#f = open("pred_statistics_slope05_approach602.csv", 'r')

#f = open("pred_statistics_slope03_approach602.csv", 'r')
#f = open("pred_statistics_slope09_approach602.csv", 'r')

#f = open("pred_statistics_slope05_approach1000.csv", 'r')
#f = open("pred_statistics_slope05_approach1500.csv", 'r')
#f = open("pred_statistics_slope05_approach2000.csv", 'r')
#f = open("pred_statistics_slope05_approach5000.csv", 'r')
f = open("pred_statistics_slope05_approach10000.csv", 'r')

#Skip the head line
f.readline()

for line in f:
    tmp = line.split(',')
    tmp = tmp[1].split(' ')
    tmp.pop()
    #print tmp
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
    print(label, 1.0*popular_labels_count[label]/count)

