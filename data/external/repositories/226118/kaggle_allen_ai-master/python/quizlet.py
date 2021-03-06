#!/usr/bin/env python
# quizlet.py for  in /home/maxime/workspace
#
# Made by Maxime MARCHES
# Login   <marche_m@epitech.eu>
#
# Started on  Thu Jan 21 22:20:15 2016 Maxime MARCHES
# Last update Tue Feb 23 13:41:36 2016 Maxime MARCHES
#

# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import urllib2
import json
import os.path
import time

debug = False
shouldSleep = False
defaultOutputFolder = "/tmp"

def getOutputFolder():
    if len(sys.argv) > 3 :
        return sys.argv[3]
    return defaultOutputFolder


def getFileName(setId):
    return '%s.json' % setId

def getFilePath(setId):
    return '%s/%s' %(getOutputFolder(), getFileName(setId))

def setExist(setId):
    return os.path.isfile(getFilePath(setId))

def publicCall(url):
    composer="?"
    if "?" in url:
        composer="&"
    url = 'https://api.quizlet.com/2.0%s%sclient_id=%s&whitespace=1' % (url, composer, sys.argv[2])
    req = urllib2.Request(url)
    res = urllib2.urlopen(req)
    return res.read()

def saveTerms(oneSet, terms):
    text_file = open(getFilePath(oneSet['id']), "w")
    text_file.write(terms)
    text_file.close()

def getSets(word, index, page):
    resp = publicCall("/search/sets?q=%s&per_page=50&page=%s" %(word, page))

    if debug:
        print resp

    obj = json.loads(resp)
    total=len(obj['sets'])
    current = 1

    for oneSet in obj['sets']:

        if not setExist(oneSet['id']):
            terms = publicCall("/sets/%s" % oneSet['id'])
            saveTerms(oneSet, terms)

        current += 1

    return resp

def getData():
    filename=sys.argv[1]

    totalLines = sum(1 for line in open(filename))

    with open(filename) as fp:
        number = 1;
        for line in fp:

            word=line.rstrip('\n')
            sys.stdout.write("%s/%s -> id:%s\r" %(number, totalLines, word))
            sys.stdout.flush()

            getSets(word, number, 1)

            if shouldSleep:
                time.sleep(0.25)

            number += 1

    return

if __name__ == "__main__":
    if len(sys.argv) < 3 :
        print "usage:", sys.argv[0], "inputFile apiToken [ouputFolder]"
        exit
    else:
        getData()
