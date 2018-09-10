# -*- coding: utf-8 -*-

"""
simple code to calculate Pearson correlation between two results
"""

import pandas as pd
import numpy as np

print('Reading data...')
xgb1 = pd.read_csv("../output/xgboost_1.csv")
xgb2 = pd.read_csv("../output/xgboost_2.csv")
rf = pd.read_csv("../output/rf.csv")
gbm = pd.read_csv("../output/gbm.csv")

print('Pearson correlation = ', np.corrcoef(gbm.Hazard, rf.Hazard)[0,1])