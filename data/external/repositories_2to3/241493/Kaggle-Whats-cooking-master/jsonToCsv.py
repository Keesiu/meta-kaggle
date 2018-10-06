import json
from pprint import pprint
fw = open('traindata.csv','w')
with open('train.json') as data_file:    
	data = json.load(data_file)

for i in range(len(data)) :

	s = str(data[i]['id']) + "\t" + str(data[i]['cuisine']) + "\t"
	ingd = data[i]['ingredients']
	for j in ingd :
		j = str([x for x in j if ord(x)>31 and ord(x)<128])
		s += str(j).strip().replace("-","") + " "
	fw.write(str(s).strip()+"\n")


