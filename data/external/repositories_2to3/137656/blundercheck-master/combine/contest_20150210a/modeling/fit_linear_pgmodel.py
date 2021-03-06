#!/usr/bin/env python

import sys, time
import numpy as np
import pickle as pickle
from pandas import DataFrame
from pandas import read_pickle
from pandas import get_dummies
import statsmodels.formula.api as sm

from djeval import *


msg("Hi, reading yy_df.")
yy_df = read_pickle(sys.argv[1])

msg("Getting subset ready.")

# TODO save the dummies along with yy_df
dummies = get_dummies(yy_df['opening_feature'])

# TODO save the moveelo_features along with yy_df
moveelo_features = [("moveelo_" + x) for x in ['mean', 'median', '25', '10', 'min', 'max', 'stdev']]

train = yy_df[yy_df.meanerror.notnull() & yy_df.elo.notnull()]

formula_rhs = "side + nmerror + gameoutcome + drawn_game + gamelength + meanecho"
formula_rhs = formula_rhs + " + opponent_nmerror + opponent_noblunders"
formula_rhs = formula_rhs + " + min_nmerror + early_lead"
formula_rhs = formula_rhs + " + q_error_one + q_error_two"
formula_rhs = formula_rhs + " + opponent_q_error_one"
formula_rhs = formula_rhs + " + mean_depth_clipped + mean_seldepth"
formula_rhs = formula_rhs + " + mean_depths_ar + mean_deepest_ar"
formula_rhs = formula_rhs + " + opponent_mean_depths_ar + opponent_mean_deepest_ar"
formula_rhs = formula_rhs + " + pct_sanemoves"
formula_rhs = formula_rhs + " + " + " + ".join(dummies.columns.values)
formula_rhs = formula_rhs + " + moveelo_weighted"

# Never mind these, they didnt help much
#formula_rhs = formula_rhs + " + " + " + ".join(moveelo_features)


formula = "elo ~ " + formula_rhs

msg("Fitting!")
ols = sm.ols(formula=formula, data=train).fit()
print(ols.summary())
