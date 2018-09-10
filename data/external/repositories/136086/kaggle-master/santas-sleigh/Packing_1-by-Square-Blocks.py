from __future__ import division, print_function
import csv

##############################################################

import itertools

def flatten(nested_lst):
    return list(itertools.chain.from_iterable(nested_lst))

def nested_index(item, nested_lst):
    for i, sublist in enumerate(nested_lst):
        try:
            j = sublist.index(item)
            return (i, j)
        except: continue
    raise ValueError(str(item) + " is not in nested list")

##############################################################

class Present:
    def __init__(self, row):
        present = list(map(int, row))
        present = [present[0]] + sorted(present[1:4])
        
        self.idx = present[0]
        self.large = present[3]
        self.median = present[2]
        self.small = present[1]
        # we always have small<median<large
        
    def __repr__(self):
        return (str(self.idx) + ': ('
                + str(self.large) + ', '
                + str(self.median) + ', '
                + str(self.small) + ')')

class Present_Position:
    def __init__(self, idx, x, y, z, dx, dy, dz):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def move(self, ddx, ddy, ddz):
        self.x += ddx
        self.y += ddy
        self.z += ddz

    def __repr__(self):
        return (str(self.idx) + ": ("
                + str(self.x) + ", "
                + str(self.y) + ", "
                + str(self.z) + "), ("
                + str(self.dx) + ", "
                + str(self.dy) + ", "
                + str(self.dz) + ")")

    def output(self):
        return [self.idx, self.x, self.y, self.z, self.x+self.dx, self.y+self.dy, self.z+self.dz]

##############################################################

presents, presents_2, presents_4 = [], [], []

with open('presents.csv', newline='\n') as csvfile:
    
    csvstream = csv.reader(csvfile, delimiter=',')
    next(csvstream) # skip the header line
    
    for row in csvstream:
        present = Present(row)

        presents.append(present)

        if present.idx <= 698904 and present.small >= 65 and present.large >= 71: presents_4.append(present)
        elif present.idx >= 698905 and present.large >= 11: presents_2.append(present)

##############################################################

def pack_small(small_presents, edge_size):
    packed = []
    small_remains = []
    x, y, z = 0, 0, 0
    done = False
    for present in small_presents:
        if not done and x+present.median <= edge_size: 
            packed.append(Present_Position(present.idx, x, y, 0, present.median, present.large, -present.small))
            x += present.median
            z = min(z, -present.small)
        else: done = True
        if done: small_remains.append(present)
    return (z, small_remains, packed)

def pack_with_large(large, small_presents, edge_size):
    packed = [Present_Position(large.idx, 0, 0, 0, large.large, large.median, -large.small)]
    x, y, z = 0, large.median, -large.small
    for present in small_presents:
        if present.median <= edge_size - y:
            packed.append(Present_Position(present.idx, x, y, 0, present.small, present.median, -present.large))
            x += present.small
            z = min(z, -present.large)
        elif present.small <= edge_size - y:
            packed.append(Present_Position(present.idx, x, y, 0, present.median, present.small, -present.large))
            x += present.median
            z = min(z, -present.large)
        else: return (False, 0, [])
    if x > edge_size: return (False, 0, [])
    else: return (True, z, packed)

##############################################################

# for the first part, we use 250*250 blocks to fill presents_4
z_level = [[0]*4, [0]*4, [0]*4, [0]*4]

idx_tofill = 0
filled_count, runout_count = 0, 0
packed = []

for large in presents_4:
    presents_buffer = presents[idx_tofill:(large.idx-1)]
    
    success = False
    while (not success):
        z_tofill = max(flatten(z_level))
        i, j = nested_index(z_tofill, z_level)
            
        success, z, packed_block = pack_with_large(large, presents_buffer, 250)
        if success:
            [x.move(i*250, j*250, z_level[i][j]) for x in packed_block]
            # The above line returns None, but we only care about the change of it given by "move"
            packed.extend(packed_block)
            z_level[i][j] += z
            
            idx_tofill = large.idx
            filled_count += 1
        else:
            z, presents_buffer, packed_block = pack_small(presents_buffer, 250)
            [x.move(i*250, j*250, z_level[i][j]) for x in packed_block]
            packed.extend(packed_block)
            z_level[i][j] += z

            runout_count += 1

    #if 0 < idx_tofill % 10000 < 5: print(idx_tofill)
    if idx_tofill > 698904: break

print("filled block #:", filled_count)
print("runout block #:", runout_count)

#print(z_level)

##############################################################

# for the second part, we use 70*70 blocks to fill presents_2

z_min = min(flatten(z_level))
z_level = [[z_min]*14, [z_min]*14, [z_min]*14, [z_min]*14, [z_min]*14,
           [z_min]*14, [z_min]*14, [z_min]*14, [z_min]*14, [z_min]*14,
           [z_min]*14, [z_min]*14, [z_min]*14, [z_min]*14]

print(z_min)

idx_tofill = 698905
filled_count, runout_count = 0, 0

for large in presents_2:
    presents_buffer = presents[idx_tofill:(large.idx-1)]
    
    success = False
    while (not success):
        z_tofill = max(flatten(z_level))
        i, j = nested_index(z_tofill, z_level)
            
        success, z, packed_block = pack_with_large(large, presents_buffer, 70)
        if success:
            [x.move(i*70, j*70, z_level[i][j]) for x in packed_block]
            # The above line returns None, but we only care about the change of it given by "move"
            packed.extend(packed_block)
            z_level[i][j] += z
            
            idx_tofill = large.idx
            filled_count += 1
        else:
            z, presents_buffer, packed_block = pack_small(presents_buffer, 70)
            [x.move(i*70, j*70, z_level[i][j]) for x in packed_block]
            packed.extend(packed_block)
            z_level[i][j] += z

            runout_count += 1

    #if 0 < idx_tofill % 10000 < 5: print(idx_tofill)

print("filled block #:", filled_count)
print("runout block #:", runout_count)

z_min = min(flatten(z_level))
print(z_min)

#filled block #: 199995
#runout block #: 8909
#-1416056
#filled block #: 150499
#runout block #: 1127
#-1432432

# so the score is = 1432432*2 = 2864864
