#!/usr/bin/env python

import sys, argparse
import ngram

PROGRESS = 500000

WORD = ngram.load_table('word')
WORD_AI = max(WORD.values()) if len(WORD) > 0 else 0
print("Loaded %d words. Starting at word id %d" \
    % (len(WORD), WORD_AI), file=sys.stderr)
def word_id(word, outfile):
    global WORD, WORD_AI
    word = word[:45]
    v = WORD.get(word, None)
    if v is None:
        WORD_AI += 1
        v = WORD_AI
        print('%d\t%s' % (v, word), file=outfile)
    return v
    
POS = ngram.load_table('pos')
POS_AI = max(POS.values()) if len(POS) > 0 else 0
print("Loaded %d POS. Starting at pos id %d" \
    % (len(POS), POS_AI), file=sys.stderr)
NGRAM_POS = set(['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 
                 'CONJ', 'PRT', 'X', '.'])
def pos_id(tag, outfile):
    global POS, POS_AI, NGRAM_POS
    if tag not in NGRAM_POS:
        raise ValueError("Not a POS tag")
    v = POS.get(tag, None)
    if v is None:
        POS_AI += 1
        v = POS_AI
        print('%d\t%s' % (v, tag), file=outfile)
    return v

def main(word_file, pos_file, ngram_file, ngram_word_file, ngram_freq_file):
    ngram_id = ngram.max_id('ngram')
    ngram.cur.close()
    ngram.db.close()
    cur_ngram = None
    total_freq = 0
    for line in sys.stdin:
        try:
            entry = line.rstrip().split('\t')
            if entry[0] != cur_ngram: # new n-gram, import words
                # Write previous ngram to file
                if cur_ngram is not None:
                    print('%d\t%d\t%d' \
                        % (ngram_id, len(words), total_freq), file=ngram_file)
                
                cur_ngram = entry[0]
                ngram_id += 1
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
                
            #year = entry[1]
            freq = int(entry[2])
            #vol = entry[3]
            #print >>ngram_freq_file, '%d\t%s\t%d\t%s' % (ngram_id, year, freq, vol)
            total_freq += freq
        except Exception as e:
            print(e, file=sys.stderr)
            print(line)
            
    # The last ngram
    print('%d\t%d\t%d' \
        % (ngram_id, len(words), total_freq), file=ngram_file)
    
def opts():
    parser = argparse.ArgumentParser(
        description='Convert ngrams to DB schema for bulk import')
    parser.add_argument('word_file', type=argparse.FileType('w'),
        help='word file')
    parser.add_argument('pos_file', type=argparse.FileType('w'),
        help='pos file')
    parser.add_argument('ngram_file', type=argparse.FileType('w'),
        help='ngram file')
    parser.add_argument('ngram_word_file', type=argparse.FileType('w'),
        help='ngram_word file')
    parser.add_argument('ngram_freq_file', type=argparse.FileType('w'),
        help='ngram_freq file')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    main(**vars(args))
