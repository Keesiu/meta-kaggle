#!/usr/bin/env python

import argparse, itertools, sys
import pickle as pickle

import numpy as np
from util import Prediction, tokenize_words
from sklearn.ensemble import RandomForestClassifier
        
def load(fd):
    d = np.load(fd)
    return d['X']

def load_classifier(fd):
    results = pickle.load(fd)
    return results['rf']

def opts():
    parser = argparse.ArgumentParser(
        description='Find optimal threshold for inserting words')
    parser.add_argument('data', type=argparse.FileType('r'),
        help='npz file with training data')
    parser.add_argument('classifier', type=argparse.FileType('r'),
        help='Input pickle file with classifier to re-use')
    parser.add_argument('predictions', type=argparse.FileType('r'),
        help='Input file with predicted words and locations')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    print("Loading test data", file=sys.stderr)
    X = load(args.data)
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X)   
 
    print("Loading classifer", file=sys.stderr)
    clf = load_classifier(args.classifier)
    
    print("Predicting decisions", file=sys.stderr)
    d = clf.predict(X)
    
    print("Performing decisions on stdin", file=sys.stderr)
    for di, line, pred in zip(d, sys.stdin, args.predictions):
        pred = Prediction.parse(pred)
        words = tokenize_words(line)
        if di == 0: #  do nothing
            pass
        elif di == 1: # insert space
            words.insert(pred.location, ' ')
        else: # insert word
            words.insert(pred.location, pred.word)
        print(' '.join(words))