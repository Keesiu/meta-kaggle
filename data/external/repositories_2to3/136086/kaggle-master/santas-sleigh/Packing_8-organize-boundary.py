
from copy import copy
import csv
import pyglet
from pyglet.gl import *
from pyglet.window import mouse

##############################################################

class Present:
    def __init__(self, row):
        present = list(map(int, row))
        present = [present[0]] + sorted(present[1:4])
        self.idx = present[0]
        self.large = present[3]
        self.median = present[2]
        self.small = present[1]
        # we always have small<=median<=large
    def __repr__(self):
        return (str(self.idx) + ': ('
                + str(self.large) + ', '
                + str(self.median) + ', '
                + str(self.small) + ')')

def classifier(present):
    if present.large >= 71 and present.small >= 65: return 4
    if present.large >= 71: return 3
    if present.large <= 10: return 1
    else: return 2

hmax = {1:10, 2:70, 3:100, 4:250}

def block_num(presents):
    i, area = 0, 0
    while (area <= 1000*1000):
        area += presents[i].small * presents[i].median
        i += 1
    return i

##############################################################

class Box:
    def __init__(self, x, y, z, dx, dy, dz):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
    def __repr__(self):
        return "((" + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + '), (' + str(self.dx) + ', ' + str(self.dy) + ', ' + str(self.dz) + "))"

class Position:
    def __init__(self, idx, classify, x, y, z, dx, dy, dz):
        self.idx = idx
        self.classify = classify
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
        return (str(self.idx) + ": (("
                + str(self.x) + ", "
                + str(self.y) + ", "
                + str(self.z) + "), ("
                + str(self.dx) + ", "
                + str(self.dy) + ", "
                + str(self.dz) + "))")
    def output(self):
        return ','.join(map(str, [self.idx,
                                  self.x+1, self.y+1, self.z, self.x+1, self.y+1, self.z+self.dz+1,
                                  self.x+self.dx, self.y+1, self.z, self.x+self.dx, self.y+1, self.z+self.dz+1,
                                  self.x+1, self.y+self.dy, self.z, self.x+1, self.y+self.dy, self.z+self.dz+1,
                                  self.x+self.dx, self.y+self.dy, self.z, self.x+self.dx, self.y+self.dy, self.z+self.dz+1])) + '\n'

##############################################################

def sort_presents(presents):
    presents_classified = [None, None, None, None, None]
    for i in range(1, 5):
        presents_classified[i] = sorted([p for p in presents if classifier(p) == i], key=lambda p: p.median, reverse=True)
    return presents_classified[4]+presents_classified[3]+presents_classified[2]+presents_classified[1]

def devide(presents, length):
    if len(presents) == 0: return ([], [], length)
    else:
        present = presents[0]
        if present.small > length:
            result = devide(presents[1:], length)
            result[1].append(present)
            return result
        else:
            result1 = devide(presents[1:], length)
            result2 = devide(presents[1:], length-present.small)
            if result2[2] <= result1[2]:
                result2[0].append(present)
                return result2
            else:
                result1[1].append(present)
                return result1

def pack_layer(presents, boxes, recycled, cut):
    boxes = (sorted([box for box in boxes if box.dx < presents[0].small or box.dy < presents[0].median or box.dy > 250 or box.dx >= int(1000/cut)],
                    key=lambda box: min(box.dx, box.dy))
             + sorted([box for box in boxes if box.dx >= presents[0].small and box.dy >= presents[0].median and box.dy <= 250 and box.dx < int(1000/cut)],
                      key=lambda box: box.dx, reverse=True))
    box = boxes.pop()
    
    height = box.dz
    if hmax[classifier(presents[0])] < height:
        recycled.append(Box(box.x, box.y, hmax[classifier(presents[0])], box.dx, box.dy, box.dz-hmax[classifier(presents[0])]))
        height = hmax[classifier(presents[0])]

    switch_l2r, switch_r2l = False, False
    gap, switch_num = 350, 7

    i, dx, dy_max  = 0, 0, 0
    packed, edge = [], []
    while (i < len(presents)):
        if (switch_l2r == False and dx > 1000-gap and len(presents) > i+switch_num and classifier(presents[i+switch_num]) == 4):
            result = devide(presents[i:(i+switch_num+1)], 1000-dx)
            presents[i:(i+switch_num+1)] = sorted(result[0], key=lambda p: p.median, reverse=True) + sorted(result[1], key=lambda p: p.median, reverse=True)
        if dx + presents[i].small <= box.dx and presents[i].median <= box.dy:
            packed.append(Position(presents[i].idx, classifier(presents[i]), box.x + dx, box.y, 0, presents[i].small, presents[i].median, -presents[i].large))
            edge.append((dx + presents[i].small, presents[i].median))
            dx += presents[i].small
            dy_max = max(dy_max, presents[i].median)
            i += 1
        else:
            boxes.append(Box(box.x+dx, box.y, 0, box.dx-dx, presents[i-1].median, height)) #side box
            break

    dy_max_l2r = dy_max
        
    if i < len(presents) and hmax[classifier(presents[i])] < height:
        recycled.append(Box(box.x, box.y+dy_max_l2r, hmax[classifier(presents[i])], box.dx, box.dy-dy_max_l2r, height-hmax[classifier(presents[i])]))
        height = hmax[classifier(presents[i])]

    dx = box.dx
    dy_edge = (box.dx, 0)
    while (i < len(presents)):
        if (switch_r2l == False and dx < gap and len(presents) > i+switch_num and classifier(presents[i+switch_num]) == 4):
            result = devide(presents[i:(i+switch_num+1)], dx)
            presents[i:(i+switch_num+1)] = sorted(result[0], key=lambda p: p.median, reverse=True) + sorted(result[1], key=lambda p: p.median, reverse=True)
        while len(edge) > 0 and edge[-1][0] >= dx - presents[i].small:
            dy_edge_tmp = edge.pop()
            dy_edge = (dy_edge_tmp[0], max(dy_edge[1], dy_edge_tmp[1]))
        if presents[i].small <= dx and dy_edge[1] + presents[i].median <= box.dy:
            packed.append(Position(presents[i].idx, classifier(presents[i]), box.x + dx - presents[i].small, box.y + dy_edge[1], 0, presents[i].small, presents[i].median, -presents[i].large))
            dx -= presents[i].small
            dy_max = max(dy_max, dy_edge[1] + presents[i].median)
            i += 1
        else:
            if box.dx == 1000 and box.dx > 3*(box.dy-dy_max):
                boxes.append(Box(box.x, box.y+dy_max, 0, int(box.dx/cut), box.dy-dy_max, height)) #top box (split left)
                boxes.append(Box(box.x+int(box.dx/cut), box.y+dy_max, 0, box.dx-int(box.dx/cut), box.dy-dy_max, height)) #top box (split right)
            else: boxes.append(Box(box.x, box.y+dy_max, 0, box.dx, box.dy-dy_max, height)) #top box
            if dx < box.dx: boxes.append(Box(box.x, box.y+dy_max_l2r, 0, dx, dy_max-dy_max_l2r, height)) #side box
            break

    if i == 0: return False
    if i == len(presents): return packed

    packed_next = pack_layer(presents[i:], boxes, recycled, cut)
    if packed_next == False: return False
    else:
        packed.extend(packed_next)
        return packed

def pack_one_in_recycled(present, recycled):
    if classifier(present) <= 3: recycled.sort(key=lambda box: box.z-0.000001*box.dx*box.dy, reverse=True)
    if classifier(present) == 4: recycled.sort(key=lambda box: box.z+0.000001*box.dx*box.dy, reverse=True)

    box = recycled.pop()

    #print(classifier(present), recycled, box)

    if box.dx > 70 and box.dy > 150:
        if present.small < box.dx and present.large < box.dy and present.median < box.dz:
            recycled.append(Box(box.x+present.small, box.y, box.z, box.dx-present.small, box.dy, box.dz))
            return Position(present.idx, classifier(present), box.x, box.y, -box.z, present.small, present.large, -present.median)
        if present.median < box.dx and present.large < box.dy and present.small < box.dz:
            recycled.append(Box(box.x+present.median, box.y, box.z, box.dx-present.median, box.dy, box.dz))
            return Position(present.idx, classifier(present), box.x, box.y, -box.z, present.median, present.large, -present.small)

    box_edges = sorted([box.dx, box.dy, box.dz])
    if present.small < box_edges[0] and present.median < box_edges[1] and present.large < box_edges[2]:
        order = sorted(list(range(3)), key=lambda k: [box.dx, box.dy, box.dz][k])
        rotation = [0, 0, 0]
        rotation[order[0]] = present.small
        rotation[order[1]] = present.median
        rotation[order[2]] = present.large
        position = Position(present.idx, classifier(present), box.x, box.y, -box.z, rotation[0], rotation[1], -rotation[2])
        if box.dx >= box.dy: recycled.append(Box(box.x+rotation[0], box.y, box.z, box.dx-rotation[0], box.dy, box.dz))
        else: recycled.append(Box(box.x, box.y+rotation[1], box.z, box.dx, box.dy-rotation[1], box.dz))
        return position
    else: return False

def pack_recycled(presents, recycled):
    di = 0
    packed = []
    while(recycled):
        if di >= len(presents): break
        position = pack_one_in_recycled(presents[di], recycled)
        if position != False: 
            packed.append(position)
            di += 1
    return di, packed

##############################################################

def show_packed(packed):
    """
    window = pyglet.window.Window(250,750)

    def drawBlock(x, y, dx, dy, classify):
        glBegin(GL_POLYGON)
        if classify == 4: glColor3f(1, 0, 0)
        elif classify == 3: glColor3f(0, 1, 0)
        elif classify == 2: glColor3f(0, 0, 1)
        else: glColor3f(1, 1, 1)
        glVertex2f((x-950)*5, (y-0)*5)
        glVertex2f((x-950)*5+dx*5, (y-0)*5)
        glVertex2f((x-950)*5+dx*5, (y-0)*5+dy*5)
        glVertex2f((x-950)*5, (y-0)*5+dy*5)
        glEnd()

    """
    window = pyglet.window.Window(500,500)

    def drawBlock(x, y, dx, dy, classify):
        glBegin(GL_POLYGON)
        if classify == 4: glColor3f(1, 0, 0)
        elif classify == 3: glColor3f(0, 1, 0)
        elif classify == 2: glColor3f(0, 0, 1)
        else: glColor3f(1, 1, 1)
        glVertex2f(x/2, y/2)
        glVertex2f(x/2+dx/2, y/2)
        glVertex2f(x/2+dx/2, y/2+dy/2)
        glVertex2f(x/2, y/2+dy/2)
        glEnd()
    #"""

    @window.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        [drawBlock(position.x, position.y, position.dx, position.dy, position.classify) for position in packed]

    pyglet.app.run()

##############################################################

presents = []

with open('presents.csv', 'rb') as csvfile:
    
    csvstream = csv.reader(csvfile, delimiter=',')
    next(csvstream) #skip the header line
    
    for row in csvstream:
        present = Present(row)
        presents.append(present)

##############################################################

"""
i = 0#47750#698905 # 0
di_max, dii_max = 0, 0
packed_layer_max = []
packed_recycled_max = []
for cut in (3, 3.25, 3.5, 3.75, 4, 4.5, 5, 6, 7, 8):
    for di in range(block_num(presents[i:]), 1, -1):
        recycled = []
        packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled, cut)
        if packed_layer != False:
            recycled.sort(key=lambda box: box.z, reverse=True)
            dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
            if di+dii > di_max+dii_max:
                #print(sort_presents(presents[i:(i+di)]))
                di_max = di
                dii_max = dii
                packed_layer_max = packed_layer
                packed_recycled_max = packed_recycled
                print(cut)
            break

print(di_max, dii_max)
show_packed(packed_recycled_max)
#show_packed(packed_layer_max)

"""
i, z = 0, 0
packed = []

while i < 399480:
    di_max, dii_max = 0, 0
    packed_layer_max = []
    packed_recycled_max = []
    for cut in (3, 3.25, 3.5, 3.75, 4, 4.5, 5, 6, 7, 8):
        for di in range(block_num(presents[i:]), 1, -1):
            recycled = []
            packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled, cut)
            if packed_layer != False:
                dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
                if di+dii > di_max+dii_max:
                    di_max = di
                    dii_max = dii
                    packed_layer_max = packed_layer
                    packed_recycled_max = packed_recycled
                    #print(cut)
                break
    for position in packed_layer_max: position.move(0, 0, z)
    for position in packed_recycled_max: position.move(0, 0, z)
    packed.extend(packed_layer_max)
    packed.extend(packed_recycled_max)
    z = min([position.z+position.dz for position in packed_layer_max]
            + [position.z+position.dz for position in packed_recycled_max])
    print(i, di_max+dii_max, di_max, dii_max, z)
    i += (di_max+dii_max)

while i < 698905-150:
    di_max, dii_max = 0, 0
    packed_layer_max = []
    packed_recycled_max = []
    for cut in (4, 5, 6, 7, 8, 9, 10):
        for di in range(block_num(presents[i:]), 1, -1):
            recycled = []
            packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled, cut)
            if packed_layer != False:
                dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
                if di+dii > di_max+dii_max:
                    di_max = di
                    dii_max = dii
                    packed_layer_max = packed_layer
                    packed_recycled_max = packed_recycled
                break
    for position in packed_layer_max: position.move(0, 0, z)
    for position in packed_recycled_max: position.move(0, 0, z)
    packed.extend(packed_layer_max)
    packed.extend(packed_recycled_max)
    z = min([position.z+position.dz for position in packed_layer_max]
            + [position.z+position.dz for position in packed_recycled_max])
    print(i, di_max+dii_max, di_max, dii_max, z)
    i += (di_max+dii_max)

if i < 698905:
    di_max, dii_max = 0, 0
    packed_layer_max = []
    packed_recycled_max = []
    for cut in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        for di in range(block_num(presents[i:]), 1, -1):
            recycled = []
            packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled, cut)
            if packed_layer != False:
                dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
                if di+dii > di_max+dii_max:
                    di_max = di
                    dii_max = dii
                    packed_layer_max = packed_layer
                    packed_recycled_max = packed_recycled
                break
    for position in packed_layer_max: position.move(0, 0, z)
    for position in packed_recycled_max: position.move(0, 0, z)
    packed.extend(packed_layer_max)
    packed.extend(packed_recycled_max)
    z = min([position.z+position.dz for position in packed_layer_max]
            + [position.z+position.dz for position in packed_recycled_max])
    print(i, di_max+dii_max, di_max, dii_max, z)
    i += (di_max+dii_max)
    print("CONNECTED!")

while i < 1000000:
    for di in range(2500, 1, -1):
        #recycled = []
        packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 500, 1000, 70), Box(500, 0, 0, 500, 1000, 70)], recycled, cut=2)
        if packed_layer != False:
            dii = 0
            for position in packed_layer: position.move(0, 0, z)
            packed.extend(packed_layer)
            z = min([position.z+position.dz for position in packed_layer] + [position.z+position.dz for position in packed_recycled])
            print(i, i+di+dii, di, dii, z)
            i += (di+dii)
            break

for position in packed: position.move(0, 0, -z)

g = open("submission.csv", 'w')
g.write("PresentId,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6,x7,y7,z7,x8,y8,z8\n")
packed.sort(key=lambda position: position.idx)
for position in packed: g.write(position.output())
g.close()
#"""
# So the score should be 1036365*2 = 2072730
# tiny improvement
# THIRD SUBMISSION
