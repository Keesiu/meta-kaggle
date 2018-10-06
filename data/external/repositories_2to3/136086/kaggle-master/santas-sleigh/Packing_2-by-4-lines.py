
from copy import copy
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

def add_present(present, packed, yz):
    if present.median < cutoff:
        packed.append(Present_Position(present.idx, 0, 0, 0, present.large, present.small, -present.median))
        yz.z = min(yz.z, -present.median)
        yz.y += present.small
    elif present.small < cutoff:
        packed.append(Present_Position(present.idx, 0, 0, 0, present.large, present.median, -present.small))
        yz.z = min(yz.z, -present.small)
        yz.y += present.median
    else:
        packed.append(Present_Position(present.idx, 0, 0, 0, present.large, present.small, -present.median))
        yz.y += present.small
        yz.y_tall += present.small
        yz.z_tall = min(yz.z_tall, -present.median)
    return yz

class YZuntil:
    def __init__(self, y = 0, z = 0, y_tall = 0, z_tall = 0):
        self.y = y
        self.z = z
        self.y_tall = y_tall
        self.z_tall = z_tall

"""
class Z:
    def __init__(self, z = 0, y_max = 1000):
        self.z = z
        self.y_max = y_max
"""
cutoff = 120

idx_tofile = 0
z_all, y_max, z_max = [0, 0, 0, 0], [1000, 1000, 1000, 1000], [0, 0, 0, 0]

yz = YZuntil(0, 0, 0, 0)
yz_undo = copy(yz)

packed = []
    
for present in presents_4:

    yz_undo = copy(yz)
    yz = add_present(present, packed, yz)

    if yz.y > y_max[idx_tofile]:

        yz = copy(yz_undo)
        packed.pop()

        idx_tofile = z_all.index(max(z_all))
        if yz.y_tall > 750:
            z_all[idx_tofile] += yz.z_tall
            y_max[idx_tofile] = 1000
            z_max[idx_tofile] = 0
        else:
            z_all[idx_tofile] += yz.z
            y_max[idx_tofile] = 1000 - yz.y_tall
            z_max[idx_tofile] = yz.z_tall - yz.z
            
        """
        print(packed)
        print("y:", yz.y)
        print("z:", yz.z)
        print("y_tall", yz.y_tall)
        print("z_tall", yz.z_tall)
        if present.idx > 100: break
        """

        idx_tofile = z_all.index(max(z_all))
        yz = YZuntil(0, z_max[idx_tofile], 0, 0)
        packed = []

        yz_undo = copy(yz)
        yz = add_present(present, packed, yz)

print("z_level", z_all)
print("y_max", y_max)
print("z_max", z_max)

#z_level [-1183899, -1183931, -1183826, -1183850]
#y_max [474, 813, 673, 836]
#z_max [-127, -74, -108, -118]

# so the score will be larger than 1183931*2 = 2367862
# not quite interesting
