#!/usr/bin/env python

import sys, argparse

PROGRESS = 100000
NGRAM_POS = set(['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 
                 'CONJ', 'PRT', 'X'])

def main():
    cur_ngram = None
    ngram_counts = []
    for line in sys.stdin:
        try:
            entry = line.strip().split('\t')
            if entry[0] != cur_ngram: # new n-gram, import words
                # Write previous ngram to file
                if cur_ngram is not None:
                    print('%d\t%d\t%d' \
                        % (ngram_id, len(words), total_freq), file=ngram_file)
                
                cur_ngram = entry[0]
                total_freq = 0
                words = entry[0].split()
                if ngram_id % PROGRESS == 0:
                    print(ngram_id, file=sys.stderr)
                
                for i, word in enumerate(words):
                    try: 
                        word, pos = word.split('_')
                        pid = pos_id(pos, pos_file)
                    except: 
                        pid = 'NULL'
                    finally:
                        wid = word_id(word, word_file)
                    print('%d\t%d\t%d\t%s' \
                        % (ngram_id, i, wid, pid), file=ngram_word_file)
                
            year = entry[1]
            freq = int(entry[2])
            vol = entry[3]
            print('%d\t%s\t%d\t%s' % (ngram_id, year, freq, vol), file=ngram_freq_file)
            total_freq += freq
        except Exception as e:
            print(e, file=sys.stderr)

if __name__ == "__main__":
    main()
