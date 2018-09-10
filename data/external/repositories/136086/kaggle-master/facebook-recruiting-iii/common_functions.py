from __future__ import division, print_function
import string, subprocess, re

######################################################

def Fmeasure(real, predicted, same):
    if same == 0 or real==0 or predicted == 0: return 0
    else: return 2 * (1.0*same/real) * (1.0*same/predicted) / (1.0*same/real + 1.0*same/predicted)

def FmeasureStr(predictedStr, realStr):
    predictedStr = set(predictedStr)
    realStr = set(realStr)
    predicted = len(predictedStr)
    real = len(realStr)
    same = len(realStr.intersection(predictedStr))
    return Fmeasure(real, predicted, same)

######################################################

def findSynonyms(word):
    synonyms = []
    if len(word) >= 4:
        if word[-1:] == 's': synonyms.append(word[:-1])
        if word[-2:] == 'ed': synonyms.append(word[:-2])
        if word[-2:] == '\'s': synonyms.append(word[:-2])
        if word[-3:] == 'ing': synonyms.append(word[:-3])
    return synonyms

def sentance2words(sentance):
    words = re.split('[ ?;,]', sentance.lower())
    for i in range(len(words)):
        words.extend(findSynonyms(words[i]))
        ## It seems that doesn't appear frequently in questions.
        #if words[i].find('-') >= 0: words.extend(words[i].split('-'))
    return filter(None, words)

def doesKeyInStr(key, sentance):
    titleTags = sentance2words(sentance)

    if key in titleTags: return True
    if len(set(findSynonyms(key)).intersection(set(titleTags))) >= 1: return True       

    """
    if len(key) >= 4:
        ## Prefix/suffix in words in sentance automatically doesn't matter.
        if sentance.lower().find(key) >= 0: return True
        
        ## Those parts handle the suffix of the keyword, which can change
        ## the F-measures of title keywords from 0.5018 to 0.5251.
        ## However, I still haven't setup a way to check those synonyms
        ## in the prediction part (if I want to keep the cost under O(log n)
        if key[-1:] == 's' and sentance.lower().find(key[:-1]) >= 0: return True
        if key[-2:] == 'ed' and sentance.lower().find(key[:-2]) >= 0: return True
        if key[-3:] == 'ing' and sentance.lower().find(key[:-3]) >= 0: return True
    else:
        if key in sentance2words(sentance): return True
    """
    
    if key.find('-') >= 0:
        #titleTags = sentance2words(sentance)
        tag = TagCompounds(key)
        if len(set(tag.compounds).intersection(titleTags)) >= tag.count: return True
    return False

def delectLanguage(sentance):
    f = open("tmp/train_code.txt", "w")
    f.write(sentance)
    f.close()
    language = subprocess.check_output("node detect_language.js tmp/train_code.txt", shell = True)[:-1]
    if language == "cs": language = "c#"
    elif language == "cpp": language = "c++"
    elif language == "fsharp": language = "f#"
    elif language == "objectivec": language = "objective-c"
    elif language == "vbnet": language = ".net"
    elif language == "cmake": language = "make"
    return language

######################################################

class TagInfo:

    def __init__(self, everybody=0, yes=0, no=0, incorrect=0, related=set()):
        self.count = everybody
        self.countYes = yes
        self.countNo = no
        self.countIncorrect = incorrect
        self.relatedTags = related

    def __repr__(self):
        lst = [self.count, self.countNo, self.relatedTags]
        return ("("
                + str(self.count) + "/"
                + str(self.countYes) + ","
                + str(self.countNo) + ","
                + str(self.countIncorrect) + ") - "
                + str(list(self.relatedTags)) + "\n")
        #return str([self.count, self.countYes, self.countNo, self.relatedTags])

    def __lt__(self, other): self.count < other.count

    def included(self):
        self.count += 1
        self.countYes += 1
    def excluded(self):
        self.count += 1
        self.countNo += 1
        
    def incorrect(self): self.countIncorrect += 1

    def addRelatedTag(self, tag): self.relatedTags.add(tag)

def renewTagsInfoIncorrect(tagsInfo, cursor):

    tags = tagsInfo.keys()

    ## Compounds will roughly double the time (when keyword number
    ## ~ 500), and it's cost is O(n) rather than O(log n).
    ## However, the improvement for F-measure is tiny.
    #tagsCompounds = {}
    #for tag in tags:
    #    if tag.find('-') >= 0: tagsCompounds[tag] = TagCompounds(tag)
    
    numrows = int(cursor.rowcount)
    for x in range(0, numrows):
        row = cursor.fetchone()
        title = row[0]
        titleTags = sentance2words(title)
        
        matchTags = set(tags).intersection(set(titleTags))

        #for k, v in tagsCompounds.iteritems():
        #    if len(set(v.compounds).intersection(titleTags)) >= v.count:
        #        matchTags.add(k)

        for tag in matchTags: tagsInfo[tag].incorrect()

    for k, v in tagsInfo.iteritems(): v.countIncorrect = v.countIncorrect - v.countYes

######################################################

class TagCompounds:

    def __init__(self, word, relatedTags=''):
        self.count = 1
        self.word = word
        self.compounds = []
        if len(word) >= 4 and word.find('-') > 0:
            tmp = word.split('-')
            self.count = len(tmp)
            self.compounds.extend(tmp)

    def __repr__(self):
        return str(self.compounds)

class TagInUse:

    def __init__(self, synonym, relatedTags=''):
        self.synonym = synonym
        self.relatedTags = relatedTags
