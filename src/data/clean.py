import os
import pandas as pd
from time import clock

t1 = clock()
count_error = 0
for dirpath, dirnames, filenames in os.walk(r'C:\Users\keesi\git-repos\meta-kaggle-2016\data\external\github-repositories', topdown=True):
    for file in filenames:
        if file[-3:] != '.py':
            try:
                os.remove(os.path.join('c:', os.sep ,os.path.normpath(dirpath), file))
            except Exception as ex:
                print("Error:", os.path.join('c:', os.sep ,os.path.normpath(dirpath), file))
                template = "An exception of type {0} occurred. Arguments:{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                count_error += 1
t2 = clock()
print("Time needed for cleaning files:", str(pd.Timedelta(seconds=t2-t1)))