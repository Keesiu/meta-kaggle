# -*- coding: utf-8 -*-

"""
code to combine data from different algorithms with different degrees and
coefficient
"""

import pandas as pd

print('Reading data...')
xgb1 = pd.read_csv("../output/xgboost_1.csv")
xgb2 = pd.read_csv("../output/xgboost_2.csv")
rf = pd.read_csv("../output/rf.csv")
gbm = pd.read_csv("../output/gbm.csv")

print('Combining...')
ind = rf.Id

d_xgb1 = 1.5
d_xgb2 = 0.4
d_rf = 0.006
d_gbm = 0.006

c_xgb1 = 0.765
c_xgb2 = 0.035
c_rf = 0.15
c_gbm = 0.15

xgb1 = xgb1.Hazard ** d_xgb1 * c_xgb1
xgb2 = xgb2.Hazard ** d_xgb2 * c_xgb2
rf = rf.Hazard ** d_rf * c_rf
gbm = gbm.Hazard ** d_gbm * c_gbm

yp = xgb1 + xgb2 + rf + gbm

print('Saving...')
preds = pd.DataFrame({"Id": ind, "Hazard": yp})
preds = preds.set_index('Id')
preds.to_csv('../output/combined_predictions.csv')