#!/usr/bin/env python

'''
Identify the location of a missing word in a sentence
using a POS-tag n-gram model. Computes gap probability
as:

P(gap m) = P(ngram) / (P(mgram) * P(lgram))
'''

import sys, argparse, pickle
import numpy as np
from util import window, tokenize_words

def find_missing_word(words, ngrams, mgrams, lgrams, n, m, l):
    if len(words) < n: 
        raise ValueError("Sentence has %d < %d tokens" % (len(words), n))
        
    gapscore = []
    for ngram in window(words, n):
        try:
            p_n = float(ngrams.get(ngram, 0))
            p_m = mgrams.get(ngram[:m], 0)
            p_l = lgrams.get(ngram[m:], 0) 
            score = p_n / (p_m * p_l)
        except: 
            score = float('inf')
        gapscore.append(score)
    print(np.log10(gapscore), file=sys.stderr)
    idx = np.argmin(gapscore) + m
    print(' '.join(words[:idx]) + ' _______ ' + ' '.join(words[idx:]), file=sys.stderr)
    return idx

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ngrams', type=argparse.FileType('r'),
        help='Pickle file with POS n-grams')
    parser.add_argument('mgrams', type=argparse.FileType('r'),
        help='Pickle file with POS m-grams (m < n)')
    parser.add_argument('lgrams', type=argparse.FileType('r'),
        help='Pickle file with POS l-grams (m + l = n)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print("Loading n-gram counts", file=sys.stderr)
    ngrams = pickle.load(args.ngrams)
    n = len(list(ngrams.keys())[0])
    print("Loading m-gram counts", file=sys.stderr)
    mgrams = pickle.load(args.mgrams)
    m = len(list(mgrams.keys())[0])
    print("Loading l-gram counts", file=sys.stderr)
    lgrams = pickle.load(args.lgrams)
    l = len(list(lgrams.keys())[0])
    if m+l != n:
        raise ValueError("m (%d) + l (%d) != n (%d)" % (m, l, n))

    for line in sys.stdin:
        try: 
            words = tokenize_words(line)
            print(find_missing_word(words, ngrams, mgrams, lgrams, n, m, l))
        except Exception as e: 
            print("ERROR: %s" % line.rstrip(), file=sys.stderr)
            print(e, file=sys.stderr)
            print(0)
