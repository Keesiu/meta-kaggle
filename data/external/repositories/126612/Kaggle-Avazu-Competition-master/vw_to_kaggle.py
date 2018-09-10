#!/usr/bin/env python

#######################################################
# File to convert vw predictions in raw form into     #
# probabilities and write to submission file suitable #  
# for kaggle                                          #
# from the Avazu click-through rate prediction contest#
# author: Jeremiah Johnson                            #
# credit: Triskelion                                  #
#######################################################                                 

import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--loc_preds", required=True, help="location of vw prediction file")
ap.add_argument("-d", "--loc_csv", required=True, help="location of output csv file")
args = vars(ap.parse_args())

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open(args["loc_csv"],"wb") as outfile:
    outfile.write("id,click\n")
    for line in open(args["loc_preds"]):
        row = line.strip().split(" ")
        outfile.write("%s,%f\n"%(row[1],zygmoid(float(row[0]))+1e-15))
