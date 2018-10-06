from statistics import *
from pattern import *
from itertools import chain

def list2grid(lst, edge):
    grid = []
    while(lst):
        grid += [lst[:edge]]
        lst = lst[edge:]
    return grid

def grid2list(grid):
    return list(chain(*grid))

def deltaStep(delta, grid):
    if (delta == 0): return grid
    else: return deltaStep(delta-1, step(grid))

def stepBack(grid, infoTree):
    
    def voteCell(grid, infoTree, i, j):
        
        def voteSub(rank, grid, infoTree, i, j):
            if (isinstance(infoTree, Info)):
                return infoTree.vote
            else:
                return voteSub(rank+1, grid, infoTree[neighbor(rank, grid, i, j)], i, j)

        return voteSub(0, grid, infoTree, i, j)

    voteGrid = [[voteCell(grid, infoTree, i, j)
                 for j in range(0, edge)] for i in range(0, edge)]
    matchPattern(grid, voteGrid)
    return voteGrid
            
def deltaBack(delta, grid, infoList):
    if (delta == 0): return grid
    else: return deltaBack(delta-1, stepBack(grid, infoList[delta]), infoList)

def difference(grid1, grid2):
    return sum(grid2list([[abs(grid1[i][j] - grid2[i][j])
                           for j in range(0, edge)] for i in range(0, edge)]))/float(edge**2)

if __name__ == '__main__':

    infoList = getInfoList()
    
    #grid = [[random.randrange(2) for j in range(0, edge)] for i in range(0, edge)]
    #print deltaBack(4, grid, infoList)
    #print difference(grid, deltaBack(1, step(grid), infoList))
