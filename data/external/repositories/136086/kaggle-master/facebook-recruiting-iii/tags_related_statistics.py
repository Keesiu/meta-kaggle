from __future__ import division, print_function
import MySQLdb, re, time
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

f = open("tmp/tags_relations1.txt", 'r') # It seems that tags_relations2.txt is basically included, so not quite useful.

tagsInfo = {}

for relation in f:
    info = re.split('[ ,()\n]', relation)
    tag, relatedTag, supp, conf = info[2], info[0], float(info[4]), float(info[6])
    
    if conf >= threshold_relatedtag:
        if tag not in tagsInfo: 
            cursor.execute("SELECT * FROM tags WHERE name = '" + tag + "';")
            row = cursor.fetchone()
            tagsInfo[tag] = TagInfo(row[1], row[2], row[3], 0, set())
            ## If leave argument related blank (although default value is also set()),
            ## the relatedTags parameter of all classes will point to the same set
            ## in python. Therefore, it doesn't work.
        tagsInfo[tag].addRelatedTag(relatedTag)

######################################################

tagsInfoCopy = {}

for k, v in tagsInfo.iteritems():
    cursor.execute("SELECT EXISTS (SELECT * FROM tags_use WHERE name = '"
                   + k
                   + "');")
    if cursor.fetchone()[0]:
        cursor.execute("UPDATE tags_use SET relatedTags = '"
                       + ' '.join(v.relatedTags)
                       + "' WHERE name = '"
                       + k
                       + "';")
    else: tagsInfoCopy[k] = v

## just filter out the keys which already exist in database
tagsInfo = tagsInfoCopy

cursor.execute("SELECT title FROM sample;")
db.commit()

renewTagsInfoIncorrect(tagsInfo, cursor)

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

print("Executed tags_related_statistics.py - Time: " + str(time.time()-startTime))
