#!/usr/bin/env python

'''
Identify the location of a missing word in a sentence
using a POS-tag n-gram model. Computes gap likelihood
as:

P(gap) = P(a, *, b) / P(a, b)
'''

import sys, argparse, pickle
from collections import defaultdict
import numpy as np
from scipy.misc import logsumexp
from util import window, tokenize_words, normalize_ngrams

def marginalize(trigrams):
    gapgrams = defaultdict(list)
    for k, v in trigrams.items():
        gapgrams[(k[0], k[2])].append(v)
    gapgrams = {k: logsumexp(v) for k, v in gapgrams.items()}
    return gapgrams

def find_missing_word(words, bigrams, gapgrams):
    if len(words) < 2: return 0
    gapscore = []
    #words = ['<s>'] + words + ['</s>']
    for ngram in window(words, 2):
        try:
            score = gapgrams[ngram] - bigrams[ngram]
        except: 
            score = float('-inf')
        gapscore.append(score)
    idx = np.argmax(gapscore) + 1
    return idx

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bigrams', type=argparse.FileType('r'),
        help='Pickle file with POS bi-grams')
    parser.add_argument('trigrams', type=argparse.FileType('r'),
        help='Pickle file with POS tri-grams')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print("Loading bi-gram counts", file=sys.stderr)
    bigrams = normalize_ngrams(pickle.load(args.bigrams))
    print("Loading tri-gram counts", file=sys.stderr)
    trigrams = normalize_ngrams(pickle.load(args.trigrams))
    print("Marginalizing tri-grams over gaps", file=sys.stderr)
    gapgrams = marginalize(trigrams)
    del trigrams

    for line in sys.stdin:
        try: 
            words = tokenize_words(line)
            print(find_missing_word(words, bigrams, gapgrams))
        except Exception as e: 
            print("ERROR: %s" % line.rstrip(), file=sys.stderr)
            print(e, file=sys.stderr)
            print(0)
