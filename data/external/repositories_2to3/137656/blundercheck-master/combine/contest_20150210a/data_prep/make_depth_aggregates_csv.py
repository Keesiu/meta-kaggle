#!/usr/bin/env python

import sys, csv
import numpy as np

depths = []
seldepths = []
depths_agreeing_ratio = {}
deepest_agree_ratio = {}
depths_agreeing_ratio[1] = []
depths_agreeing_ratio[-1] = []
deepest_agree_ratio[1] = []
deepest_agree_ratio[-1] = []

def movenum_to_side(movenum):
    if movenum % 2 == 0:
        return 1
    else:
        return -1

def dump_rows():
    global depths
    global seldepths
    global depths_agreeing_ratio
    global deepest_agree_ratio

    if current_game == 0:
        return
    for side in [1, -1]:
        print(current_game, end=' ')
        print(side, end=' ')
        print(np.mean(depths), end=' ')
        print(np.mean(seldepths), end=' ')
        print(np.mean(depths_agreeing_ratio[side]), end=' ')
        print(np.mean(deepest_agree_ratio[side]), end=' ')
        if len(depths_agreeing_ratio[side]) == 0:
            print(0.5, end=' ')
        else:
            print(float(np.count_nonzero(depths_agreeing_ratio[side])) / len(depths_agreeing_ratio[side]), end=' ')
        print(len(depths))
    depths = []
    seldepths = []
    depths_agreeing_ratio[1] = []
    depths_agreeing_ratio[-1] = []
    deepest_agree_ratio[1] = []
    deepest_agree_ratio[-1] = []

columns = [
'halfply',
'moverscore',
'movergain',
'move_piece',
'move_dir',
'move_dist',
'move_is_capture',
'move_is_check',
'bestmove_piece',
'bestmove_dir',
'bestmove_dist',
'bestmove_is_capture',
'bestmove_is_check',
'depth',
'seldepth',
'depths_agreeing',
'deepest_agree',
'elo',
'side',
'gamenum',
]

depths = []
seldepths = []
depths_agreeing_ratio = {}
depths_agreeing_ratio[1] = []
depths_agreeing_ratio[-1] = []
deepest_agree_ratio = {}
deepest_agree_ratio[1] = []
deepest_agree_ratio[-1] = []

csvreader = csv.DictReader(sys.stdin, fieldnames=columns)


current_game = 0
rownum = 0

for row in csvreader:
    if row['gamenum'] != current_game:
        dump_rows()
        current_game = row['gamenum']
    depths.append(int(row['depth']))
    seldepths.append(int(row['seldepth']))
    side = int(row['side'])
    depths_agreeing_ratio[side].append(float(row['depths_agreeing']) / float(row['depth']))
    deepest_agree_ratio[side].append(float(row['deepest_agree']) / float(row['depth']))
    rownum = rownum + 1
#    if rownum > 10000:
#        break
