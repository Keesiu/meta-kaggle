from __future__ import division, print_function
import MySQLdb, time
from common_functions import *
from parameters import *

startTime = time.time()

######################################################

db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="",
                     db="facebook3")

cursor = db.cursor()

######################################################

cursor.execute("SELECT * FROM tags ORDER BY countYes DESC;")
db.commit()

tagsInfo = {}

numrows = number_popular_tag
for i in range(0, numrows):
    row = cursor.fetchone()
    tagsInfo[row[0]] = TagInfo(row[1], row[2], row[3])

######################################################

cursor.execute("SELECT title FROM sample;")
db.commit()

renewTagsInfoIncorrect(tagsInfo, cursor)
#print(tagsInfo)

######################################################

cursor.execute("TRUNCATE TABLE tags_use;")
db.commit()

for k, v in tagsInfo.iteritems():
    if v.countYes >= threshold_newtag * (v.countYes + v.countIncorrect):
        cursor.execute("INSERT INTO tags_use VALUES ('"
                       + k + "',"
                       + str(v.count) + ","
                       + str(v.countYes) + ","
                       + str(v.countNo) + ","
                       + str(v.countIncorrect) + ",'"
                       + ' '.join(v.relatedTags) + "');")

db.commit()
db.close()

######################################################

print("Executed tags_popular_statistics.py - Time: " + str(time.time()-startTime))
