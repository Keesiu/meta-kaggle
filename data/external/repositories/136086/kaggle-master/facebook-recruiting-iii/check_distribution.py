from __future__ import division, print_function
import MySQLdb, math
import matplotlib.pyplot as plt
from common_functions import *

######################################################

db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="",
                     db="facebook3")

cursor = db.cursor()

######################################################

cursor.execute("SELECT * FROM tags ORDER BY countYes DESC;")
db.commit()

tagsInfo = []

numrows = int(cursor.rowcount)
for i in range(0, numrows):
    row = cursor.fetchone()
    tagsInfo.append((row[0], TagInfo(row[1], row[2], row[3])))

countYes = filter(None, [tag[1].countYes for tag in tagsInfo])
print(len(countYes))
#16270 ## The number of tags with non-zero "countYes" values

plt.plot(map(math.log, countYes[:12000]))
plt.xlabel('tag number ranked by existance in title')
plt.ylabel('log (time appeared as a title-word)')
plt.show()
## So from the graph, 4200 first tags ~ countYes >= 5

## As there are 248130 questions in "Sample.csv",
## the support of tags for countYes >= 5 is 5/248130=0.0000201.
## Use this number to setup the "aproiri" support argument.
