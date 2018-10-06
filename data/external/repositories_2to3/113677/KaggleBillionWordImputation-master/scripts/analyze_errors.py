#!/usr/bin/env python

'''
Insert missing word predictions in predicted location
'''

import sys, argparse
import numpy as np

from util import tokenize_words, Prediction

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('golden', type=argparse.FileType('r'),
        help='File with original sentences')
    parser.add_argument('i_removed', type=argparse.FileType('r'),
        help='File with true location of removed word')
    parser.add_argument('predicted', type=argparse.FileType('r'),
        help='Predictions for missing word and location')
    return parser

def print_error(p, g, gi):
    g[gi] = '***%s***' % g[gi]
    insertions = [gi]
    g.insert(p.locations[0]+int(p.locations[0]>gi), '^^^%s(%.3f/%.3f)^^^' \
        % (p.word, p.word_posterior, p.location_posterior))
    insertions.append(p.locations[0])
    for l, pr in zip(p.locations, p.p_anywhere[1:]):
        il = l + sum(i<l for i in insertions)
        g.insert(il, '###%.3f###' % (10**(pr-p.Z)))
        insertions.append(il)
    print(' '.join(g))

if __name__ == "__main__":
    args = opts().parse_args()
    
    print("Loading golden sentences", file=sys.stderr)
    golden = list(map(tokenize_words, args.golden))
    print("Loading locations of removed words", file=sys.stderr)
    golden_loc = np.asarray(list(map(int, args.i_removed)))
    print("Loading predictions", file=sys.stderr)
    predictions = list(map(Prediction.parse, args.predicted))
    assert len(golden) == len(golden_loc)
    if len(predictions) < len(golden):
        n = len(predictions)
        golden = golden[:n]
        golden_loc = golden_loc[:n]
        print("Assuming first %d sentences" % n, file=sys.stderr)
        
    print("Processing sentences", file=sys.stderr)
    for p, g, gi in zip(predictions, golden, golden_loc):
        if p.locations[0] != gi or p.word != g[gi]: # error
            print_error(p, g, gi)
        
