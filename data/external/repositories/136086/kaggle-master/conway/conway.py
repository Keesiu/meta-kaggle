from __future__ import division, print_function
from random import randrange

edge = 20

def step(grid):
    
    def neighbor(grid, i, j):
        neighborCells = []
        if (i != edge-1):
            neighborCells.append(grid[i+1][j])
            if (j != 0): neighborCells += [grid[i][j-1], grid[i+1][j-1]]
            if (j != (edge-1)): neighborCells += [grid[i][j+1], grid[i+1][j+1]]
        if (i != 0):
            neighborCells.append(grid[i-1][j])
            if (j != 0): neighborCells.append(grid[i-1][j-1])
            if (j != (edge-1)): neighborCells.append(grid[i-1][j+1])
        return sum(neighborCells)

    def cellstep(cell, neighborSum):
        if (cell==1 and neighborSum>=2 and neighborSum<=3): return 1
        if (cell==0 and neighborSum==3): return 1
        else: return 0
    
    return [[cellstep(grid[i][j], neighbor(grid, i, j))
             for j in range(0,edge)] for i in range(0,edge)]

if __name__ == '__main__':

    import pyglet
    from pyglet.gl import *
    from pyglet.window import mouse

    size = 30
    window = pyglet.window.Window(size*edge, size*edge)

    grid = [[randrange(2) for j in range(0, edge)] for i in range(0, edge)]

    def drawBlock(i, j, size):
        glBegin(GL_POLYGON)
        glVertex2f(i*size, j*size)
        glVertex2f((i+1)*size, j*size)
        glVertex2f((i+1)*size, (j+1)*size)
        glVertex2f(i*size, (j+1)*size)
        glEnd()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        global grid
        grid = step(grid)

    @window.event
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        [[drawBlock(i, j, size) for j in range(0,edge) if (grid[i][j]==0)]
         for i in range(0,edge)]

    pyglet.app.run()
