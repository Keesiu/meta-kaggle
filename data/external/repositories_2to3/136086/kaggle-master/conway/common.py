from conway import *

def take(grid, i, j):
    if (i < 0 or i > edge-1 or j < 0 or j > edge-1): return 0
    else: return grid[i][j]

def fill(grid, i, j, numFill):
    if (i < 0 or i > edge-1 or j < 0 or j > edge-1): return
    else: grid[i][j] = numFill

def takerange(grid, i, j, di, dj):
    return [[take(grid, ii, jj) for jj in range(j, j+dj)] for ii in range(i, i+di)]

def fillrange(grid, i, j, di, dj, pattern):
    for ii in range(di):
        for jj in range(dj): fill(grid, i+ii, j+jj, pattern[ii][jj])
    return

def binlst2str(lst):
    return ''.join(map(str, lst))

def str2binlst(str):
    return list(map(int, str))

def binlst2bin(lst):
    return int(binlst2str(lst), 2)

if __name__ == '__main__':
    print((binlst2str([1,0,1,0,1,1,0])))
    print((str2binlst("10110110")))

    grid = [[randrange(2) for j in range(edge)] for i in range(edge)]
    print(grid)
    print((takerange(grid, -1, -1, 4, 4)))
    fillrange(grid, -1, -1, 4, 4, [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    print(grid)
