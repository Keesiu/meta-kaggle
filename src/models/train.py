# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_validate
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import pickle
from sklearn.preprocessing import StandardScaler

def main(processed_path = "data/processed",
         models_path = "models",
         y_name = 'ranking_log'):
    
    """Nested 10-fold cross-validation for linear regression of
    ranking_log and score with with lasso regularization
    (inner CV for alpha tuning, outer for R^2 robustness)."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    models_path = os.path.normpath(models_path)
    logger.debug("Path to models normalized: {}"
                 .format(models_path))
    
    # load selected_df
    selected_df = pd.read_pickle(os.path.join(processed_path, 'selected_df.pkl'))
    logger.info("Loaded selected_df. Shape of df: {}"
                .format(selected_df.shape))
    
    # split df into dependent and independent variables
    teams_df = selected_df.iloc[:, :9]
    y = selected_df.iloc[:, 9:11]
    X = selected_df.iloc[:, 11:]
    X_columns = X.columns
    X_index = X.index
    
    # set y to either ranking_log or score
    y = y[y_name]
    logger.info("Set y to '{}'.".format(y_name))
    
    #%% standardize
    
    scaler = StandardScaler()
    not_standardize = ['core',
                       'visualization',
                       'machine_learning',
                       'deep_learning']
    X_standardized = scaler.fit_transform(X
                                          .drop(columns=not_standardize)
                                          .values)
    X_standardized = pd.DataFrame(X_standardized,
                                  index = X.index,
                                  columns = X.columns.drop(not_standardize))
    X_not_standardized = X[not_standardize]
    X = pd.concat([X_standardized, X_not_standardized], axis=1)
    logger.info("After Standardization:\n{}".format(X.describe()))
    
    #%% define hyperparameter
    
    start = time()

#    # define list of 100 alphas to test: from 1 logarithmically decreasing to 0
#    BASE = 1 + 1/5
#    logger.debug("Constant BASE is set to {}.".format(BASE))
#    ALPHAS = [BASE**(-x) for x in range(100)]
#    ALPHAS = np.logspace(0.00001, 1, num=50, base=10.0)
#    n_samples = len(y)
#    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
#                 (n_samples * l1_ratio))
#    np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
#                       num=n_alphas)[::-1]
#    logger.debug("Alphas set to {}".format(ALPHAS))
    L1_RATIOS = [1.0, .95, .7, .5, .3, .1]
    EPS = 0.001
    N_ALPHAS = 100
    ALPHAS = None
    # normalize data
    # If True, the regressors X will be normalized before regression by
    # subtracting the mean (column-wise) and dividing by the l2-norm in
    # order for each feature to have norm = 1.
    NORMALIZE = False
    MAX_ITER = 1000
    TOL = 0.0001
    CV = 10
    N_JOBS = 1
    RS = 1
    SELECTION = 'cyclic'
    
    logger.info("l1_ratio={}, eps={}, n_alphas={}, alphas={}, normalize={}, max_iter={}, tol={}, cv={}, n_jobs={}, random_state={}, selection={}"
                 .format(L1_RATIOS, EPS, N_ALPHAS, ALPHAS, NORMALIZE,
                         MAX_ITER, TOL, CV, N_JOBS, RS, SELECTION))
    logger.debug("Try following L1-ratios: {}".format(L1_RATIOS))
    
#    # print R^2 values for bounding alphas 0 and 1 to make sense of alphas
#    logger.info("Bounding score: R^2 for alpha=0 and l1_ratio=0.5: {}"
#                .format(ElasticNet(alpha=0.000000001, l1_ratio=.5,
#                                   normalize=NORMALIZE, random_state=42)
#                        .fit(X.values, y.values)
#                        .score(X.values, y.values)))
#    logger.info("Bounding score: R^2 for alpha=1 and l1_ratio=0.5: {}"
#                .format(ElasticNet(alpha=1, l1_ratio=.5,
#                                   normalize=NORMALIZE, random_state=42)
#                        .fit(X.values, y.values)
#                        .score(X.values, y.values)))
    
    #%% train model
    
    mod = ElasticNetCV(l1_ratio = L1_RATIOS,
                       eps = EPS,
                       n_alphas = N_ALPHAS,
                       alphas = ALPHAS,
                       normalize = NORMALIZE,
                       max_iter = MAX_ITER,
                       tol = TOL,
                       cv = CV,
                       n_jobs = N_JOBS,
                       random_state = RS,
                       selection = SELECTION)\
          .fit(X.values, y.values)
    
    # log some statistics
    logger.info("best R^2 score: {}".format(mod.score(X.values, y.values)))
    best_l1_ratio = mod.l1_ratio_
    logger.info("best l1_ratio: {}".format(best_l1_ratio))
    best_alpha = mod.alpha_
    logger.info("best alpha: {}".format(best_alpha))
    logger.debug("tested alphas:\n{}".format(mod.alphas_))
    coef = pd.Series(data=mod.coef_, index=X_columns)
    logger.debug("best coefficients:\n{}".format(coef))
    
    #%% Nested Cross-Validation to test robustness of R^2
    
    cv_results = cross_validate(ElasticNetCV(l1_ratio = L1_RATIOS,
                                             eps = EPS,
                                             n_alphas = N_ALPHAS,
                                             alphas = ALPHAS,
                                             normalize = NORMALIZE,
                                             max_iter = MAX_ITER,
                                             tol = TOL,
                                             cv = CV,
                                             n_jobs = N_JOBS,
                                             random_state = RS,
                                             selection = SELECTION),
                                X.values, y.values, cv=CV,
                                return_train_score=True, n_jobs=N_JOBS)
    logger.info("95% confidence intervall: {:.2f} +/- {:.2f} (mean +/- 2*std)"
                .format(cv_results['test_score'].mean(),
                        cv_results['test_score'].std()*2))
    logger.debug("Nested cross-validation results:\n{}"
                .format(pd.DataFrame(data=cv_results)))
    
    #%% Elastic Net regression with statsmodels for summary
    
    mod_sm = sm.OLS(y.values, sm.add_constant(pd.DataFrame(data=X.values,
                                                    columns=X_columns,
                                                    index=X_index)))\
          .fit_regularized(method='elastic_net',
                           alpha=best_alpha,
                           L1_wt=best_l1_ratio,
                           refit=True)
    res = mod_sm.summary().as_text()
    logger.info("ElasticNet regression (from sm) of selected_df with respect to '{}' with alpha={} and L1_wt={}:\n{}"
                .format(y_name, best_alpha, best_l1_ratio, res))
    
    # Normality of residuals
    # Jarque-Bera test:
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(mod_sm.resid)
    lzip(name, test)
    # Omni test:
    name = ['Chi^2', 'Two-tail probability']
    test = sms.omni_normtest(mod_sm.resid)
    lzip(name, test)
    
    # Multicollinearity
    # Conditional Number:
    np.linalg.cond(mod_sm.model.exog)
    
    # Heteroskedasticity tests
    # Breush-Pagan test:
    name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
    test = sms.het_breuschpagan(mod_sm.resid, mod_sm.model.exog)
    lzip(name, test)
    # Goldfeld-Quandt test
    name = ['F statistic', 'p-value']
    test = sms.het_goldfeldquandt(mod_sm.resid, mod_sm.model.exog)
    lzip(name, test)
    
    # Linearity
    # Harvey-Collier multiplier test for Null hypothesis that
    # the linear specification is correct:
    name = ['t value', 'p value']
    test = sms.linear_harvey_collier(mod_sm)
    lzip(name, test)
    
    #%% export results as pickle file to models folder
    
    # pickle mod
    with open(os.path.join(models_path, y_name+'_sklearn_ElasticNetCV.pkl'), 'wb') as handle:
        pickle.dump(mod, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved elastic net model of sklearn to {}."
                .format(os.path.join(models_path, y_name+'_sklearn_ElasticNetCV.pkl')))
    
    # pickle mod_sm
    with open(os.path.join(models_path, y_name+'_sm_OLS_fit_regularized.pkl'), 'wb') as handle:
        pickle.dump(mod_sm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved elastic net model of statsmodels to {}."
                .format(os.path.join(models_path, y_name+'_sm_OLS_fit_regularized.pkl')))
    
    # save res as .txt
    f = open(os.path.join(models_path, y_name+'_sm_OLS_fit_regularized_summary.txt'), "w+")
    f.write(res)
    f.close()
    
    
    #%% logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to train Elastic Net Model on selected_df with respect to '{}': {}"
                .format(y_name, time_passed))
    
#%%
if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/train.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Trains models and saves them to <models_path>.")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to load the selected data selected_df.pkl \
                    (default: data/processed)")
    parser.add_argument(
            '--models_path',
            default = "models",
            help = "path to save the trained models \
                    (default: models)")
    parser.add_argument(
            '--y_name',
            default = 'ranking_log',
            help = "name of dependent variable to be trained on \
                    either 'ranking_log' or 'score' \
                    (default: ranking_log)")
    args = parser.parse_args()
    
    # run main
    main(args.processed_path, args.models_path, args.y_name)
