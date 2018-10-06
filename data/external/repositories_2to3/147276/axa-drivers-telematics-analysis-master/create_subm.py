
import sys
import subprocess
import os

if __name__ == "__main__":
    folder = sys.argv[1]
    files = os.listdir(folder)
    filenames = [idd for idd in files if idd[0] != "." and idd[-4:] == ".csv" and len(idd) < 9]
    with open('%s/submission.csv' % folder, 'w') as outfile:
        outfile.write("driver_trip,prob\n")
        for fname in filenames:
            with open("%s/%s" % (folder,fname)) as infile:
                for line in infile:
                    outfile.write(line)
