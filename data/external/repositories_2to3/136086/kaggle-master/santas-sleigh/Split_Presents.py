
import csv

###############################################################

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

presents, presents_1, presents_2, presents_3, presents_4 = [], [], [], [], []

#with open('presents_sample.csv', newline='\n') as csvfile:
with open('presents.csv', newline='\n') as csvfile:
    
    csvstream = csv.reader(csvfile, delimiter=',')
    next(csvstream) # skip the header line
    
    for row in csvstream:
        present = Present(row)

        presents.append(present)
        
        if present.small >= 65 and present.large >= 71: presents_4.append(present)
        elif present.large >= 71: presents_3.append(present)
        elif present.large <= 10: presents_1.append(present)
        else: presents_2.append(present)
