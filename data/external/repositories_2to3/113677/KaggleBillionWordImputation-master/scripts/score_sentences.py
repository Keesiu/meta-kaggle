#!/usr/bin/env python

import sys, argparse
import kenlm

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',
        help='KenLM n-gram model file (ARPA or binary)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    print("Loading language model", file=sys.stderr)
    model = kenlm.LanguageModel(args.model)
    
    print("Processing sentences", file=sys.stderr)
    for line in sys.stdin:
        sentence = line.rstrip()
        score = model.score(sentence)
        print("P = %.3f: %s" % (score, sentence))