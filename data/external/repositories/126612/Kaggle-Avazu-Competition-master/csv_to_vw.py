#!/usr/bin/env python

########################################################
# munge csv to vw                                      #
# separate namespace for every var                     #
# hour feature split into hr (0 - 23) and date         # 
# author: Jeremiah Johnson                             #
# credit: Triskelion                                   #
########################################################

from datetime import datetime
from csv import DictReader
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--loc_csv", required=True, help="path to csv")
ap.add_argument("-d", "--loc_output", required=True, help="destination path")
ap.add_argument("-t", "--train", required = False, default=False, help ="is this training data?", action ='store_true')
args = vars(ap.parse_args()) 

start = datetime.now()
print("\nTurning %s into %s. Is train set? %s"%(args["loc_csv"],args["loc_output"],args["train"]))
  
with open(args["loc_output"],"wb") as outfile:
  for e, row in enumerate( DictReader(open(args["loc_csv"])) ):

    if args["train"]: # we care about labels
      if row['click'] == "1":
        label = 1
      else: 
        label = -1 # set negative label to -1 in vw format
      outfile.write( "%s '%s " % (label, row['id']))
        
      for k, v in row.items():
          if k == "hour":
              hr = int(v) % 100
              date = (int(v) - hr) / 100
              outfile.write("|hr %s |date %s" % (hr, date))
          if k not in ["hour", "click", "id"]:
              outfile.write("|%s %s " % (k, v))
      outfile.write("\n")

    else: # test, we don't care about labels
      outfile.write("1 '%s " % row['id'])
      for k, v in row.items():
          if k == "hour":
              hr = int(v) % 100
              date = (int(v) - hr) / 100
              outfile.write("|hr %s |date %s" % (hr, date))
          if k not in ["hour", "click", "id"]:
              outfile.write("|%s %s " % (k, v))
      outfile.write("\n")

      # Reporting progress
    if e % 1000000 == 0:
      print("%s\t%s"%(e, str(datetime.now() - start)))

print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

# usage: for training data
# python csv_to_vw.py -l path_to_train_csv -d path_to_train_output_vw -t 
# for test data, no -t flag
# python csv_to_vw.py -l path_to_test_csv -d path_to_test_output_vw
