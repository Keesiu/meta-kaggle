from __future__ import division, print_function
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

def classifier (present):
    if present.large >= 71 and present.small >= 65: return 4
    if present.large >= 71: return 3
    if present.large <= 10: return 1
    else: return 2

hmax = {1:10, 2:70, 3:100, 4:250}

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

def pack_layer(presents, boxes, recycled):
    boxes = sorted(boxes, key=lambda box: min(box.dx, box.dy))
    box = boxes.pop()
    
    height = box.dz
    if hmax[classifier(presents[0])] < height:
        recycled.append(Box(box.x, box.y, hmax[classifier(presents[0])], box.dx, box.dy, box.dz-hmax[classifier(presents[0])]))
        height = hmax[classifier(presents[0])]

    i, dx, dy_max  = 0, 0, 0
    packed, edge = [], []
    dy_start = presents[i].median
    while (i < len(presents)):
        if dx + presents[i].small <= box.dx and presents[i].median <= box.dy:
            packed.append(Position(presents[i].idx, classifier(presents[i]), box.x + dx, box.y, 0, presents[i].small, presents[i].median, -presents[i].large))
            edge.append((dx + presents[i].small, presents[i].median))
            dx += presents[i].small
            dy_max = max(dy_max, presents[i].median)
            i += 1
        else:
            boxes.append(Box(box.x+dx, box.y, 0, box.dx-dx, presents[i-1].median, height)) #side box
            break

    dx = box.dx
    dy_edge = (box.dx, 0)
    while (i < len(presents)):
        while len(edge) > 0 and edge[-1][0] >= dx - presents[i].small: dy_edge = edge.pop()
        if presents[i].small <= dx and dy_edge[1] + presents[i].median <= box.dy:
            packed.append(Position(presents[i].idx, classifier(presents[i]), box.x + dx - presents[i].small, box.y + dy_edge[1], 0, presents[i].small, presents[i].median, -presents[i].large))
            dx -= presents[i].small
            dy_max = max(dy_max, dy_edge[1] + presents[i].median)
            i += 1
        else:
            if box.dx == 1000 and box.dx > 3*(box.dy-dy_max):
                boxes.append(Box(box.x, box.y+dy_max, 0, int(box.dx/2), box.dy-dy_max, height))
                boxes.append(Box(box.x+int(box.dx/2), box.y+dy_max, 0, int(box.dx/2), box.dy-dy_max, height)) #top box (split)
            else: boxes.append(Box(box.x, box.y+dy_max, 0, box.dx, box.dy-dy_max, height)) #top box
            if dx < box.dx: boxes.append(Box(box.x, box.y+dy_start, 0, dx, dy_max-dy_start, height)) #side box
            break

    if i == 0: return False
    if i == len(presents): return packed

    packed_next = pack_layer(presents[i:], boxes, recycled)
    if packed_next == False: return False
    else:
        packed.extend(packed_next)
        return packed

def pack_one_in_recycled(present, recycled):
    box = recycled.pop()
    box_edges = sorted([box.dx, box.dy, box.dz])
    #print("present", present)
    #print("box", box)
    if present.small < box_edges[0] and present.median < box_edges[1] and present.large < box_edges[2]:
        order = sorted(range(3), key=lambda k: [box.dx, box.dy, box.dz][k])
        rotation = [0, 0, 0]
        rotation[order[0]] = present.small
        rotation[order[1]] = present.median
        rotation[order[2]] = present.large
        position = Position(present.idx, classifier(present), box.x, box.y, box.z, rotation[0], rotation[1], rotation[2])
        #print(position)
        #print("fit~~")
        if box.dx >= box.dy: recycled.append(Box(box.x+rotation[0], box.y, box.z, box.dx-rotation[0], box.dy, box.dz))
        else: recycled.append(Box(box.x, box.y+rotation[1], box.z, box.dx, box.dy-rotation[1], box.dz))
        return position
    else: return False

def pack_recycled(presents, recycled):
    di = 0
    packed = []
    while(recycled):
        position = pack_one_in_recycled(presents[di], recycled)
        if position != False: 
            packed.append(position)
            di += 1
            if di >= len(presents): break
    return di, packed

##############################################################

def show_packed(packed):
    """
    window = pyglet.window.Window(500,800)

    def drawBlock(x, y, dx, dy, classify):
        glBegin(GL_POLYGON)
        if classify == 4: glColor3f(1, 0, 0)
        elif classify == 3: glColor3f(0, 1, 0)
        elif classify == 2: glColor3f(0, 0, 1)
        else: glColor3f(1, 1, 1)
        glVertex2f((x-950)*10, (y-120)*10)
        glVertex2f((x-950)*10+dx*10, (y-120)*10)
        glVertex2f((x-950)*10+dx*10, (y-120)*10+dy*10)
        glVertex2f((x-950)*10, (y-120)*10+dy*10)
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
recycled = []
i = 0#698905 # 0
for di in range(250, 1, -1):
    packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled)
    if packed_layer != False: break
recycled.sort(key=lambda box: box.z, reverse=True)
print(recycled)
show_packed(packed_layer)

#di, packed_recycled = pack_recycled(presents[15:], recycled)
#print(di, packed_recycled)
#print(recycled)

#print(presents[14])
"""

#"""
i, z = 0, 0
packed = []
while i < 698905-50:
    recycled = []
    for di in range(250, 1, -1):
        packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled)
        recycled.sort(key=lambda box: box.z, reverse=True)
        if packed_layer != False:
            dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
            for position in packed_layer: position.move(0, 0, z)
            for position in packed_recycled: position.move(0, 0, z)
            packed.extend(packed_layer)
            packed.extend(packed_recycled)
            z = min([position.z+position.dz for position in packed_layer] + [position.z-position.dz for position in packed_recycled])
            print(i, i+di+dii, di, dii, z)
            i += (di+dii)
            break

if i < 698905:
    recycled = []
    for di in range(2500, 1, -1):
        packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 250)], recycled)
        recycled.sort(key=lambda box: box.z, reverse=True)
        if packed_layer != False:
            dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
            for position in packed_layer: position.move(0, 0, z)
            for position in packed_recycled: position.move(0, 0, z)
            packed.extend(packed_layer)
            packed.extend(packed_recycled)
            z = min([position.z+position.dz for position in packed_layer] + [position.z-position.dz for position in packed_recycled])
            print(i, i+di+dii, di, dii, z)
            i += (di+dii)
            break
    print("CONNECTED!")

while i < 1000000:
    recycled = []
    for di in range(2500, 1, -1):
        packed_layer = pack_layer(sort_presents(presents[i:(i+di)]), [Box(0, 0, 0, 1000, 1000, 70)], recycled)
        recycled.sort(key=lambda box: box.z, reverse=True)
        if packed_layer != False:
            dii, packed_recycled = pack_recycled(presents[(i+di):], recycled)
            for position in packed_layer: position.move(0, 0, z)
            for position in packed_recycled: position.move(0, 0, z)
            packed.extend(packed_layer)
            packed.extend(packed_recycled)
            z = min([position.z+position.dz for position in packed_layer] + [position.z-position.dz for position in packed_recycled])
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
# So the score should be a little bit larger than 1066277*2 = 2132554
# tiny improvement
