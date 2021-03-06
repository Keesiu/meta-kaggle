import csv
import sys
import copy
import os

SZ = 130401

if len(sys.argv) < 2:
    print("Usage: Input_folder out.csv")
    exit(1)
path = sys.argv[1]

fo = csv.writer(open(sys.argv[2], "w"), lineterminator='\n')
files = os.listdir(path)
fis = [csv.reader(file(path + x)) for x in files]

head = ""
for fi in fis:
    head = next(fi)
fo.writerow(head)


sz = 121
for i in range(SZ):
    data = []
    for j in range(len(fis)):
        fi = fis[j]
        data.append(next(fi))
    tag = [data[0][0]]
    value = [0] * sz
    for row in data:
        for j in range(1, len(row)):
            value[j - 1] += float(row[j])
    for j in range(len(value)):
        value[j] /= len(fis) * 1.0
    tag.extend(value)
    fo.writerow(tag)




