#!/usr/bin/env python

'''
Compare POS tags to a gold standard.
'''

import sys, argparse, pickle
from collections import defaultdict

from util import tokenize_words, pos_tag

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=argparse.FileType('r'),
        help='POS-tagged sentences')
    parser.add_argument('removed', type=argparse.FileType('r'),
        help='File with indices of removed words in gold')
    parser.add_argument('gold', type=argparse.FileType('r'),
        help='Gold-standard POS-tagged sentences')
    parser.add_argument('errors', type=argparse.FileType('w'),
        help='Pickle file with errors broken down by POS tag')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    counts = defaultdict(lambda: defaultdict(int))
    nerrors = 0
    nsentences = 0
    
    for sentence, ref_sentence, i_removed in zip(args.sample, args.gold, args.removed):
        try:
            i_removed = int(i_removed)
            words = tokenize_words(sentence)
            ref_words = tokenize_words(ref_sentence)
            assert len(words) == len(ref_words)-1
            pos = list(map(pos_tag, words))
            ref_pos = list(map(pos_tag, ref_words))
        
            has_error = False
            for i in range(i_removed):
                counts[pos[i]][ref_pos[i]] += 1
                has_error |= (pos[i] != ref_pos[i])
            for i in range(i_removed, len(words)):
                counts[pos[i]][ref_pos[i+1]] += 1
                has_error |= (pos[i] != ref_pos[i+1])
            if has_error: nerrors += 1
            nsentences += 1
        except Exception as e:
            print("Error processing: %s" % e, file=sys.stderr)
            sys.stderr.write(ref_sentence)
            print(sentence, file=sys.stderr)
            
    print("Found %d/%d sentences with POS-tag errors" \
        % (nerrors, nsentences), file=sys.stderr)
                
    pickle.dump(dict(counts), args.errors)
