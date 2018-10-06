
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

cursor.execute("TRUNCATE TABLE sample")
db.commit()

f = open("Sample.csv", 'r')
question = ""
FmeasureNum = []

## Skip the first line
f.readline()

for tmp in f:
    question += tmp.replace("\\", "\\\\")
    
    if question[-2:] == '\r\n':

        items = question[1:-3].split('","', 2)
        number = items[0]
        title = items[1]

        itemss = items[2].rsplit('","', 1)
        body = itemss[0]
        tagsString = itemss[1]
        tags = tagsString.split(' ')

        titleTags = " ".join([tag for tag in tags if doesKeyInStr(tag, title)])

        ##--------------------------------------------------------------

        ## Find what language the code is using
        """
        tmpcodes = question.split("</code></pre>")
        language = []
        for tmpcode in tmpcodes:
            code = tmpcode.split("<pre><code>")[1:]
            if len(code) == 1: language.append(delectLanguage(code[0]))
        languages = " ".join(language)
        """
        languages = ""
        
        cursor.execute("INSERT INTO sample VALUES ("
                       + number + ",'"
                       + re.sub('[<>_%\\\'\"]', '', title) + "','"
                       + titleTags + "','"
                       + languages + "','"
                       + tagsString + "');")
        question = ""

        FmeasureNum.append(FmeasureStr(tags, titleTags.split(' ')))

db.commit()
db.close()
f.close()

######################################################

print("Executed sample2db.py - Time: " + str(time.time()-startTime)
      + "  (F-measure: " + str(mean(FmeasureNum)) + ")")
