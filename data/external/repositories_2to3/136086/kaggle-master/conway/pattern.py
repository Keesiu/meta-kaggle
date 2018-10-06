from common import *

pattern = {'0000011001100000': '0000011001100000'}

blinker = [[0,0,0],
           [1,1,1],
           [0,0,0]]

blinkerT = list(map(list, list(zip(*blinker))))

boats = [[[0,1,0],
          [1,0,1],
          [0,1,0]],
         [[0,1,1],
          [1,0,1],
          [0,1,0]],
         [[0,1,0],
          [1,0,1],
          [0,1,1]],
         [[0,1,0],
          [1,0,1],
          [1,1,0]],
         [[1,1,0],
          [1,0,1],
          [0,1,0]],
         [[0,1,1],
          [1,0,1],
          [1,1,0]],
         [[1,1,0],
          [1,0,1],
          [0,1,1]]]

beehive = [[0,1,1,0],
           [1,0,0,1],
           [0,1,1,0]]

beehiveT = list(map(list, list(zip(*beehive))))

def matchPattern(grid, voteGrid):
    for i in range(18):
        for j in range(18):
            if (i != 0 and sum(grid[i-1][max(0,(j-1)):(j+5)]) != 0): pass
            
            if ((i == 17 or sum(grid[i+3][max(0,(j-1)):(j+5)]) == 0)
                and (j == 0 or grid[i][j-1]+grid[i+1][j-1]+grid[i+2][j-1] == 0)):
                
                if (j == 17 or grid[i][j+3]+grid[i+1][j+3]+grid[i+2][j+3] == 0):
                    if takerange(grid, i, j, 3, 3) == blinker:
                        fillrange(voteGrid, i, j, 3, 3, blinkerT)
                        pass
                    if takerange(grid, i, j, 3, 3) == blinkerT:
                        fillrange(voteGrid, i, j, 3, 3, blinker)
                        pass
                    for boat in boats:
                        if takerange(grid, i, j, 3, 3) == boat:
                            fillrange(voteGrid, i, j, 3, 3, boat)
                            pass
                    
                if (j != 17
                    and (j == 16 or (grid[i][j+4]+grid[i+1][j+4]+grid[i+2][j+4] == 0
                                     and (i == 0 or grid[i-1][j+4] == 0)
                                     and (i == 17 or grid[i+3][j+4] == 0)))):
                    if takerange(grid, i, j, 3, 4) == beehive:
                        fillrange(voteGrid, i, j, 3, 4, beehive)
                        pass

            if (i != 17
                and (i == 16 or sum(grid[i+4][max(0,(j-1)):(j+5)]) == 0)
                and (j == 0 or grid[i][j-1]+grid[i+1][j-1]+grid[i+2][j-1]+grid[i+3][j-1] == 0)
                and (j == 17 or grid[i][j+3]+grid[i+1][j+3]+grid[i+2][j+3]+grid[i+3][j+3] == 0)):
                if (takerange(grid, i, j, 4, 3) == beehiveT):
                    fillrange(voteGrid, i, j, 4, 3, beehiveT)
"""    for i in range(-1, 16):
        for j in range(-1, 15):
            if ((i == -1 or sum(grid[i][j:(j+6)]) == 0)
                and (i == 16 or sum(grid[i+4][j:(j+6)]) == 0)
                and takerange(grid, i, j, 5, 6) == beehive):
                fillrange(voteGrid, i, j, 5, 6, beehive)
    for i in range(-1, 15):
        for j in range(-1, 16):
            if ((i == -1 or sum(grid[i][j:(j+5)]) == 0)
                and (i == 15 or sum(grid[i+5][j:(j+5)]) == 0)
                and takerange(grid, i, j, 6, 5) == beehiveT):
                fillrange(voteGrid, i, j, 6, 5, beehiveT)
    return"""
