#!/usr/bin/env python

import sys, time
import numpy as np
import pickle as pickle
from pandas import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn import tree
import pygraphviz as pgv
from io import StringIO
from djeval import *

n_estimators = 200
n_cv_groups = 3
n_jobs = -1

msg("Hi, reading yy_df.")
yy_df = read_pickle(sys.argv[1])
yy_df = yy_df.set_index(['gamenum', 'side'], drop=False)

msg("Getting subset ready.")

train = yy_df[yy_df.meanerror.notnull() & yy_df.elo.notnull()]

features = list(yy_df.columns.values)
categorical_features = ['opening_feature', 'timecontrols']
excluded_features = ['elo', 'opponent_elo', 'elo_advantage', 'elo_avg', 'winner_elo_advantage', 'ols_error', 'gbr_prediction', 'gbr_error', 'ols_prediction']
excluded_features.extend(categorical_features)
for f in excluded_features:
    features.remove(f)

X = train[features].values
y = train['elo']

#print features
gamenum_index = features.index('gamenum')
side_index = features.index('side')
#ols_index = features.index('ols_prediction')

rfr = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, min_samples_leaf=10, min_samples_split=50, verbose=1)

def ols_preds_for_Xs(X):
    ols_preds = []
    for row in X:
        gamenum = int(row[gamenum_index])
        side = int(row[side_index])
#        print row
#        print "HI %i %i" % (gamenum, side)
        ols_preds.append(yy_df.loc[(gamenum, side)]['ols_prediction'])
    return np.array(ols_preds)

def blended_scorer(estimator, X, y):
    ols_preds = ols_preds_for_Xs(X)
    pred_y = estimator.predict(X)
    msg("BLENDED SCORES FOR a CV GROUP:")
    for blend in np.arange(0, 1.01, 0.1):
        blended_prediction = (blend * ols_preds) + ((1.0 - blend) * pred_y)
        blended_score = mean_absolute_error(blended_prediction, y)
        msg("%f * OLS yields score of %f" % (blend, blended_score))
    return mean_absolute_error(y, pred_y)

msg("CROSS VALIDATING DJ STYLE")
cvs = cross_val_score(rfr, X, y, cv=n_cv_groups, n_jobs=n_jobs, scoring=blended_scorer)
print(cvs)
sys.stdout.flush()

msg("CROSS VALIDATING SK STYLE TO DOUBLECHECK")
cvs = cross_val_score(rfr, X, y, cv=n_cv_groups, n_jobs=-1, scoring='mean_absolute_error')
print(cvs)
sys.stdout.flush()


msg("Fitting!")
rfr.fit(X, y)

msg("Saving model")
joblib.dump([rfr, features], sys.argv[2])

msg("Making predictions for all playergames")
yy_df['rfr_prediction'] = rfr.predict(yy_df[features].values)
yy_df['rfr_error'] = (yy_df['rfr_prediction'] - yy_df['elo']).abs()
insample_scores = yy_df.groupby('training')['rfr_error'].agg({'mean' : np.mean, 'median' : np.median, 'stdev': np.std})
print(insample_scores)

msg("Error summary by ELO:")
elo_centuries = cut(yy_df['elo'], 20)
print(yy_df.groupby(elo_centuries)['rfr_error'].agg({'sum': np.sum, 'count': len, 'mean': np.mean}))

msg("Writing yy_df back out with predictions inside")
yy_df.to_pickle(sys.argv[1])

msg("Preparing Kaggle submission")
# map from eventnum to whiteelo,blackelo array

predictions = {}
for eventnum in np.arange(25001,50001):
  predictions[eventnum] = [0,0]

for row in yy_df[yy_df['elo'].isnull()][['gamenum', 'side', 'rfr_prediction']].values:
  eventnum = row[0]
  side = row[1]
  if side == 1:
    sideindex = 0
  else:
    sideindex = 1
  prediction = row[2]
  predictions[eventnum][sideindex] = prediction

submission = open('/data/submission_rfr.csv', 'w')
submission.write('Event,WhiteElo,BlackElo\n')
for eventnum in np.arange(25001,50001):
  submission.write('%i,%i,%i\n' % (eventnum, predictions[eventnum][0], predictions[eventnum][1]))
submission.close()

print("Feature importances:")
print(DataFrame([rfr.feature_importances_, features]).transpose().sort([0], ascending=False))

print("There are %i trees." % len(rfr.estimators_))

dot_data = StringIO()
tree.export_graphviz(rfr.estimators_[0].tree_, out_file=dot_data, feature_names=features)

B=pgv.AGraph(dot_data.getvalue())
B.layout('dot')
B.draw('/data/rfr.png') # draw png

print("Wrote first tree to /data/rfr.png")
