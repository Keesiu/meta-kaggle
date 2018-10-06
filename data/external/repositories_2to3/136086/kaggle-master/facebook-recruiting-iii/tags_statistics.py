
import MySQLdb, operator, time
from common_functions import *

startTime = time.time()

######################################################

db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="",
                     db="facebook3")

cursor = db.cursor()

######################################################

cursor.execute("SELECT titleTags, tags FROM sample")
db.commit()

tagCount = [0]*10
tags = {}

numrows = int(cursor.rowcount)
for i in range(0, numrows):
    row = cursor.fetchone()
    
    titleTags = row[0].split(' ')
    realTags = row[1].split(' ')
    
    tagCount[len(realTags)] += 1
    
    for tag in realTags:
        if tag not in tags: tags[tag] = TagInfo()
        if tag in titleTags: tags[tag].included()
        else: tags[tag].excluded()

## the distribution for the tag numbers per question
## [0, 17337, 33227, 35732, 23913, 14768, 0, 0, 0, 0]
#print(tagCount)

## the most popular tags and their statistics
taglist = [(k,v) for k,v in list(tags.items())]
taglist.sort(key=(lambda x: operator.attrgetter('count')(x[1])))
taglist.reverse()
#print(taglist[:30])

######################################################

cursor.execute("TRUNCATE TABLE tags")
db.commit()

#for k, v in tags.iteritems():
for tag in taglist:
    k = tag[0]
    v = tag[1]
    cursor.execute("INSERT INTO tags VALUES ('"
                   + k + "',"
                   + str(v.count) + ","
                   + str(v.countYes) + ","
                   + str(v.countNo) + ");")

db.commit()
db.close()

######################################################
    
print("Executed tags_statistics.py - Time: " + str(time.time()-startTime))
