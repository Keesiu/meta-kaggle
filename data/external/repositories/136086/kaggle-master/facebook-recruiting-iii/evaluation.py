from __future__ import division, print_function
import MySQLdb, time
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

tags = {} # Can find/select the keywords in O(log n) time
tagsCompounds = {} # Can only be find in a for loop: O(n) time

cursor.execute("SELECT name, relatedTags FROM tags_use")
db.commit()

numrows = int(cursor.rowcount)
for i in range(0, numrows):
    row = cursor.fetchone()
    for word in [row[0]]+findSynonyms(row[0]):
        ## "key" is the synonym, element "synonym" in class
        ## "TagInUse" is the real tag.
        tags[word] = TagInUse(row[0], row[1])
    if row[0].find('-') >= 0: tagsCompounds[row[0]] = TagCompounds(row[0])

## Special cases
tags["rails"] = TagInUse("ruby-on-rails")

##----------------------------------------------------

tagsBody = set() # Can only be find in a for loop: O(n) time

cursor.execute("SELECT name FROM tags_body")
db.commit()

numrows = int(cursor.rowcount)
for i in range(0, numrows):
    row = cursor.fetchone()
    tagsBody.add(row[0])

######################################################

f = open("Evaluation.csv", 'r')
question = ""
FmeasureNum = []

i = 0
tagDistribution = [0]*20

for tmp in f:
    question += tmp.replace("\\", "\\\\")
    
    if question[-2:] == '\r\n':
        
        i += 1
        if i > 10000: break

        items = question[1:-3].split('","', 2)
        number = items[0]
        title = items[1]

        itemss = items[2].rsplit('","', 1)
        body = itemss[0]
        tagsString = itemss[1]
        realTags = tagsString.split(' ')

        titleTags = set(sentance2words(title))

        body = re.sub(re.compile(r'(<pre><code>).*(</code></pre>)', re.DOTALL), r'\1\2', body)
        body = re.sub(re.compile(r'(<blockquote>).*(</blockquote>)', re.DOTALL), r'\1\2', body)
        body = body.lower()
        
        ##--------------------------------------------------------------

        ## If match the title tags 
        predictedSynonyms = set(tags.keys()).intersection(titleTags)
        predictedTags = set([tags[x].synonym for x in predictedSynonyms])

        ## If match the title as an compound tag
        for k, v in tagsCompounds.iteritems():
            if len(set(v.compounds).intersection(titleTags)) >= v.count:
                predictedTags.add(k)

        
        ## If the tag is in the body of the question
        for tagb in tagsBody:
            if body.find(tagb.replace("-", " ")) >= 0: predictedTags.add(tagb)

        ## If is the related tag of the known predictedTags
        relatedTags = set()
        for tag in predictedTags:
            relatedTags = set(filter(None, tags[tag].relatedTags.split(' ')))
        predictedTags = predictedTags.union(relatedTags)
        
        ## If can't match anything, then just use the five most popular tags
        if not predictedTags:
            #print(title, realTags)
            predictedTags = set(["c#", "java", "javascript", "php", "android"])

        print(i, title, list(predictedTags), realTags)
        FmeasureNum.append(FmeasureStr(predictedTags, realTags))

        tagDistribution[len(predictedTags)] += 1

        question = ""

db.close()
f.close()

######################################################

print("Executed evaluation.py - Time: " + str(time.time()-startTime)
      + "  (F-measure: " + str(mean(FmeasureNum)) + ")")
print("Tag length distribution in evaluation.py: " + str(tagDistribution))

## Executed evaluation.py - Time: 134.205157042  (F-measure: 0.34076549372)
## It should be the reasonable rough time (for 150MB csv), since most of the
## times are costed because of reading the csv file, and matching the keywords.
