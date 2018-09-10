from __future__ import division, print_function
import MySQLdb, re, string, subprocess, time
from common_functions import *
from numpy import mean

startTime = time.time()

######################################################

db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="",
                     db="facebook3")

cursor = db.cursor()

######################################################

cursor.execute("TRUNCATE TABLE train")
db.commit()

f = open("Train.csv", 'r')
question = ""

#i = 0

## Skip the first line
f.readline()

for tmp in f:
    question += tmp.replace("\\", "\\\\")
    
    if question[-2:] == '\r\n':

        #i += 1
        #if i > 10000: break

        items = question[1:-3].split('","', 2)
        number = items[0]
        title = items[1]

        itemss = items[2].rsplit('","', 1)
        tagsString = itemss[1]

        ##--------------------------------------------------------------

        try:
            cursor.execute("INSERT INTO train VALUES ("
                           + number + ",'"
                           + re.sub('[<>_%\\\'\"]', '', title[:39]+'t') + "','"
                           + tagsString + "');")
        except:
            print('---------------------------')
            print(question)
        
        question = ""

db.commit()
db.close()
f.close()

######################################################

print("Executed train2db.py - Time: " + str(time.time()-startTime))
#Executed train2db.py - Time: 1908.99081993

## should have "6034195" rows from in "tail -10 Train.csv", but 2 missing.
## However, row number changes from doing "SHOW TABLE STATUS FROM facebook3;". Wierd.
