
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

## It is too slow to use FIND to find out whether that question
## exists in the training set. So I need to first read it out.

existQuestions = {}

cursor.execute("SELECT title, tags FROM train")
db.commit()

numrows = int(cursor.rowcount)
for i in range(0, numrows):
    row = cursor.fetchone()
    existQuestions[row[0]] = row[1]

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

f = open("Test.csv", 'r')
g = open("Submission.csv", 'w')
g.write('"Id","Tags"\n')

question = ""
inTrain, notInTrain = 0, 0

## Skip the first line
f.readline()

#i = 0

for tmp in f:
    question += tmp.replace("\\", "\\\\")
    
    if question[-2:] == '\r\n':
        
        #i += 1
        #if i > 100: break
        
        items = question[1:-3].split('","', 2)
        number = items[0]
        title = items[1]
        body = items[2]
        
        ##--------------------------------------------------------------

        try:
            titleKnown = re.sub('[<>_%\\\'\"]', '', title[:39]+'t')
            tagsKnown = existQuestions[titleKnown]
            g.write(str(items[0]) + ",\"" + tagsKnown + "\"\n")
            inTrain += 1

        except:
#            try:
            titleTags = set(sentance2words(title))
                    
            body = re.sub(re.compile(r'(<pre><code>).*(</code></pre>)', re.DOTALL), r'\1\2', body)
            body = re.sub(re.compile(r'(<blockquote>).*(</blockquote>)', re.DOTALL), r'\1\2', body)
            body = body.lower()
#            except:
#                print('---------------------------')
#                print(question)

            ## If match the title tags 
            predictedSynonyms = set(tags.keys()).intersection(titleTags)
            predictedTags = set([tags[x].synonym for x in predictedSynonyms])

            ## If match the title as an compound tag
            for k, v in tagsCompounds.items():
                if len(set(v.compounds).intersection(titleTags)) >= v.count:
                    predictedTags.add(k)

            ## If the tag is in the body of the question
            for tagb in tagsBody:
                if body.find(tagb.replace("-", " ")) >= 0: predictedTags.add(tagb)

            ## If is the related tag of the known predictedTags
            relatedTags = set()
            for tag in predictedTags:
                relatedTags = set([_f for _f in tags[tag].relatedTags.split(' ') if _f])
            predictedTags = predictedTags.union(relatedTags)
            
            ## If can't match anything, then just use the five most popular tags
            if not predictedTags:
                predictedTags = set(["c#", "java", "javascript", "php", "android"])

            #print(i, list(predictedTags), realTags)
            g.write(str(items[0]) + "," + '"' + " ".join(predictedTags) + '"\n')
            notInTrain += 1

        question = ""

db.close()
f.close()
g.close()

######################################################

print("Executed test.py - Time: " + str(time.time()-startTime) + "\n"
      + "In/out of the training set: " + str(inTrain) + "/" + str(notInTrain))

#Executed evaluation.py - Time: 1970.04095507
#In/out of the training set: 1167453/845884
