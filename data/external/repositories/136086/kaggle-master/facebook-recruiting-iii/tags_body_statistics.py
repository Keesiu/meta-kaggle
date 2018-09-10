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

cursor.execute("SELECT name FROM tags ORDER BY countAll DESC;")
db.commit()

tagsInfo = {}

## This number cannot be so large, because "evaluation.py" or
## "test.py" can only scan it by O(n) time.
numrows = number_body_tag
for i in range(0, numrows):
    row = cursor.fetchone()
    tagsInfo[row[0]] = TagInfo(0, 0, 0, 0)

######################################################

f = open("Sample.csv", 'r')
question = ""

## Skip the first line
f.readline()

for tmp in f:
    question += tmp.replace("\\", "\\\\")
    
    if question[-2:] == '\r\n':

        items = question[1:-3].split('","', 2)
        itemss = items[2].rsplit('","', 1)
        body = itemss[0]
        tagsString = itemss[1]
        tags = tagsString.split(' ')
            
        ## The regular expression is to remove all 
        ## codes, blockquotes in the "body" string.
        body = re.sub(re.compile(r'(<pre><code>).*(</code></pre>)', re.DOTALL), r'\1\2', body)
        body = re.sub(re.compile(r'(<blockquote>).*(</blockquote>)', re.DOTALL), r'\1\2', body)
        body = body.lower()

        ##--------------------------------------------------------------

        for tag in tagsInfo:
            if body.find(tag.replace("-", " ")) >= 0:
                if tag in tags: tagsInfo[tag].included()
                else: tagsInfo[tag].incorrect()
            else:
                if tag in tags: tagsInfo[tag].excluded()

        question = ""

######################################################

cursor.execute("TRUNCATE TABLE tags_body;")
db.commit()

for k, v in tagsInfo.iteritems():
    if (v.countYes >= threshold_countYes_body
        and v.countYes >= threshold_newtag * (v.countYes + v.countIncorrect)):
        cursor.execute("INSERT INTO tags_body VALUES ('"
                       + k + "',"
                       + str(v.count) + ","
                       + str(v.countYes) + ","
                       + str(v.countNo) + ","
                       + str(v.countIncorrect) + ");")

db.commit()
db.close()
f.close()

######################################################

print("Executed tags_body_statistics.py - Time: " + str(time.time()-startTime))
#Executed tags_body_statistics.py - Time: 36.8703811169

## roughly half of the tags survived from the threshold filters.
