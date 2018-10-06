
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

##############################################################

class Box:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
    def __repr__(self):
        return str(self.x) + ', ' + str(self.y) + ', ' + str(self.dx) + ', ' + str(self.dy)

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
        return (str(self.idx) + ": ("
                + str(self.x) + ", "
                + str(self.y) + ", "
                + str(self.z) + "), ("
                + str(self.dx) + ", "
                + str(self.dy) + ", "
                + str(self.dz) + ")")
    def output(self):
        return ','.join(map(str, [self.idx,
                                  self.x+1, self.y+1, self.z, self.x+1, self.y+1, self.z+self.dz+1,
                                  self.x+self.dx, self.y+1, self.z, self.x+self.dx, self.y+1, self.z+self.dz+1,
                                  self.x+1, self.y+self.dy, self.z, self.x+1, self.y+self.dy, self.z+self.dz+1,
                                  self.x+self.dx, self.y+self.dy, self.z, self.x+self.dx, self.y+self.dy, self.z+self.dz+1])) + '\n'

def pack_layer(presents, boxes):
    boxes = sorted(boxes, key=lambda box: min(box.dx, box.dy))
    box = boxes.pop()

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
            boxes.append(Box(box.x+dx, box.y, box.dx-dx, presents[i-1].median))
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
            boxes.append(Box(box.x, box.y+dy_max, box.dx, box.dy-dy_max))
            if dx < box.dx: boxes.append(Box(box.x, box.y+dy_start, dx, dy_max-dy_start))
            break

    if i == 0: return False
    if i == len(presents): return packed

    packed_next = pack_layer(presents[i:], boxes)
    if packed_next == False: return False
    else:
        packed.extend(packed_next)
        return packed

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
    next(csvstream) # skip the header line
    
    for row in csvstream:
        present = Present(row)
        presents.append(present)

##############################################################

#packed = pack_layer(sorted(presents[0:226], key=lambda present: present.median, reverse=True), [Box(0, 0, 1000, 1000)])
#print(packed)
#show_packed(packed)

i, z = 0, 0
packed = []
while i < 698905-50:
    for di in range(250, 1, -1):
        packed_layer = pack_layer(sorted(presents[i:(i+di)], key=lambda present: present.median, reverse=True), [Box(0, 0, 1000, 1000)])
        if packed_layer != False:
            for position in packed_layer: position.move(0, 0, z)
            packed.extend(packed_layer)
            z -= max([present.large for present in presents[i:(i+di)]])
            print(i, i+di, di, z)
            i += di
            break

if i < 698905:
    for di in range(2500, 1, -1):
        packed_layer = pack_layer(sorted(presents[i:(i+di)], key=lambda present: present.median, reverse=True), [Box(0, 0, 1000, 1000)])
        if packed_layer != False:
            for position in packed_layer: position.move(0, 0, z)
            packed.extend(packed_layer)
            z -= max([present.large for present in presents[i:(i+di)]])
            print(i, i+di, di, z)
            i += di
            break
    print("CONNECTED!")

while i < 1000000:
    for di in range(2500, 1, -1):
        packed_layer = pack_layer(sorted(presents[i:(i+di)], key=lambda present: present.median, reverse=True), [Box(0, 0, 1000, 1000)])
        if packed_layer != False:
            for position in packed_layer: position.move(0, 0, z)
            packed.extend(packed_layer)
            z -= max([present.large for present in presents[i:(i+di)]])
            print(i, i+di, di, z)
            i += di
            break

for position in packed: position.move(0, 0, -z)

g = open("submission.csv", 'w')
g.write("PresentId,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6,x7,y7,z7,x8,y8,z8\n")
packed.sort(key=lambda position: position.idx)
for position in packed: g.write(position.output())
g.close()

# So the score should be roughly 1067026*2 = 2134052 
# Submission 1
