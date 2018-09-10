import csv

map_month_date = (0, 31, 59, 90, 120, 151,
                  181, 212, 243, 273, 304, 334) 

class Toy:
    def __init__(self, row):
        self.toy_id = int(row[0])
        self.duration = int(row[2])
        self.date_time = tuple(map(int, row[1].split(" ")))
        self.minutes_from_day = self.date_time[4] + self.date_time[3]*60
        self.minutes_from_year = self.date_time[4] + self.date_time[3]*60 + \
                                 (self.date_time[2]-1+map_month_date[self.date_time[1]-1])*24*60
    def __repr__(self):
        return "(" + str(self.toy_id) + ", " + str(self.minutes_from_year) + ", " + \
        str(self.duration) + ")"

def load_toys_rev(filename):
    toy_list = []
    with open(filename, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #next(spamreader)
        for row in spamreader:
            if row[0] != "ToyId":
                toy_list.append(Toy(row))
    return toy_list
