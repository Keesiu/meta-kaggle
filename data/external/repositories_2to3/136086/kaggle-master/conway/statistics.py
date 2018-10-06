from conway import *
from common import *

threshold = 0.07
    
class Info:
    
    def __init__(self):
        self.numLive = 0
        self.numDie = 0
        self.vote = None
    def __repr__(self):
        return (str(self.vote) + ":" +
                str(self.numLive) + "/" + str(self.numDie))
    
    def live(self): self.numLive += 1
    def die(self): self.numDie += 1
    def addInfo(self, cell):
        if cell==1: self.live()
        else: self.die()
    def voteNeighbor(self):
        if (self.numLive + self.numDie > 0):
            portion = float(self.numLive)/float(self.numLive + self.numDie)
            if (portion < threshold): self.vote = 0
            elif (portion > 1-threshold): self.vote = 1
    def voteFinal(self):
        if (self.numLive <= self.numDie+1): self.vote = 0
        else: self.vote = 1

def neighbor(rank, grid, i, j):

    if (rank == 0): return grid[i][j]
    if (rank == 1):
        return binlst2bin([take(grid, i, j+1), take(grid, i+1, j),
                           take(grid, i, j-1), take(grid, i-1, j)])
    if (rank == 2):
        return binlst2bin([take(grid, i-1, j+1), take(grid, i+1, j+1),
                           take(grid, i+1, j-1), take(grid, i-1, j-1)])
    if (rank == 3):
        return binlst2bin([take(grid, i, j+2), take(grid, i+2, j),
                           take(grid, i, j-2), take(grid, i-2, j)])
    if (rank == 4):
        return sum([take(grid, i-1, j+2), take(grid, i+1, j+2),
                    take(grid, i+2, j-1), take(grid, i+2, j+1),
                    take(grid, i-1, j-2), take(grid, i+1, j-2),
                    take(grid, i-2, j-1), take(grid, i-2, j+1)])

def createInfoList(rank):
    
    def create(num): return [Info() for i in range(num)]
    
    if (rank == 0): return create(2)
    if (rank == 1): return create(16)
    if (rank == 2): return create(16)
    if (rank == 3): return create(16)
    if (rank == 4): return create(9)

def spanInfoList(rank, infoList):
    
    def spanInfoSub(recur, infoTree):
        for i in range(len(infoTree)):
            if (isinstance(infoTree[i], Info) and infoTree[i].vote != None): continue
            elif (recur == 1): infoTree[i] = createInfoList(rank)
            else: spanInfoSub(recur-1, infoTree[i])
            
    for delta in range(1, 6):
        if (rank == 0): infoList[delta] = createInfoList(rank)
        else: spanInfoSub(rank, infoList[delta])

def statisticsOneGame(rank, infoList):
    
    def statisticsCell(recur, infoTree, grid, stepgrid, i, j):
        if (isinstance(infoTree, Info) and infoTree.vote != None): return
        elif (recur == 0):
            infoTree[neighbor(rank, stepgrid, i, j)].addInfo(grid[i][j])
        else:
            statisticsCell(recur-1, infoTree[neighbor(rank-recur, stepgrid, i, j)], grid, stepgrid, i, j)
        
    grid = [[randrange(2) for j in range(edge)] for i in range(edge)]
    
    for delta in range(-4, 1): grid = step(grid)
    
    for delta in range(1, 6):
        stepgrid = step(grid)
        for i in range(0, edge):
            for j in range(0, edge):
                statisticsCell(rank, infoList[delta], grid, stepgrid, i, j)
        grid = stepgrid

def voteInfoList(rank, infoList):
    def voteSub(recur, infoTree):
        if (isinstance(infoTree, Info) and infoTree.vote != None): return
        elif (recur == 0):
            for node in infoTree: node.voteNeighbor()
        else:
            for node in infoTree: voteSub(recur-1, node)

    for delta in range(1, 6): voteSub(rank, infoList[delta])

def voteFinalList(infoList):
    
    def voteSub(infoTree):
        if isinstance(infoTree, Info):
            if (infoTree.vote == None): infoTree.voteFinal()
        else:
            for node in infoTree: voteSub(node)

    for delta in range(1, 6): voteSub(infoList[delta])

def getInfoList():

    infoList = [None for delta in range(0, 6)]

    spanInfoList(0, infoList)
    for i in range(800): statisticsOneGame(0, infoList)
    voteInfoList(0, infoList)

    spanInfoList(1, infoList)
    for i in range(800): statisticsOneGame(1, infoList)
    voteInfoList(1, infoList)

    spanInfoList(2, infoList)
    for i in range(800): statisticsOneGame(2, infoList)
    voteInfoList(2, infoList)

    spanInfoList(3, infoList)
    for i in range(800): statisticsOneGame(3, infoList)
    voteInfoList(3, infoList)

    #It seems that rank 4 will make things worse. I don't know why.
    #spanInfoList(4, infoList)
    #for i in range(10): statisticsOneGame(4, infoList)
    #voteInfoList(4, infoList)

    voteFinalList(infoList)
    return infoList

if __name__ == '__main__':

    infoList = getInfoList()
    print((infoList[1]))
