# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_validate
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import pickle
from sklearn.preprocessing import StandardScaler

def main(processed_path = "data/processed",
         models_path = "models"):
    
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
    selected_df = pd.read_pickle(os.path.join(processed_path,
                                              'selected_df.pkl'))
    logger.info("Loaded selected_df. Shape of df: {}"
                .format(selected_df.shape))
    
    #%% split df into dependent and independent variables
    teams_df = selected_df.iloc[:, :9]
    y = selected_df.iloc[:, 9:10]
    X = selected_df.iloc[:, 10:]
    X_columns = X.columns
    X_index = X.index
    
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
    logger.debug("After Standardization:\n{}".format(X.describe().to_string))
    
    #%% define hyperparameter
    
    start = time()

    L1_RATIOS = [1.0, .95, .7, .5, .3, .1]
    EPS = 0.001
    N_ALPHAS = 100
    ALPHAS = None
    # normalize data
    # If True, the regressors X will be normalized before regression by
    # subtracting the mean (column-wise) and dividing by the l2-norm in
    # order for each feature to have norm = 1.
    NORMALIZE = False
    MAX_ITER = 10000
    TOL = 0.0001
    CV = 20
    N_JOBS = 1
    RS = 1
    SELECTION = 'cyclic'
    
    logger.info("l1_ratio={}, eps={}, n_alphas={}, alphas={}, normalize={}"
                 .format(L1_RATIOS, EPS, N_ALPHAS, ALPHAS, NORMALIZE))
    logger.info("max_iter={}, tol={}, cv={}, n_jobs={}, rs={}, selection={}"
                 .format(MAX_ITER, TOL, CV, N_JOBS, RS, SELECTION))
    logger.debug("Try following L1-ratios: {}".format(L1_RATIOS))
    
    # print R^2 values for bounding alphas 0 and 1 to make sense of alphas
    logger.info("Bounding score: R^2 for alpha=0 and l1_ratio=0.5: {}"
                .format(ElasticNet(alpha=0, l1_ratio=.5,
                                   normalize=NORMALIZE, random_state=RS)
                        .fit(X.values, y.values)
                        .score(X.values, y.values)))
    logger.info("Bounding score: R^2 for alpha=1 and l1_ratio=0.5: {}"
                .format(ElasticNet(alpha=1, l1_ratio=.5,
                                   normalize=NORMALIZE, random_state=RS)
                        .fit(X.values, y.values)
                        .score(X.values, y.values)))
    
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
    best_r2 = mod.score(X.values, y.values)
    logger.info("best R^2 score: {:.2f}%".format(best_r2*100))
    best_l1_ratio = mod.l1_ratio_
    logger.info("best l1_ratio: {}".format(best_l1_ratio))
    best_alpha = mod.alpha_
    logger.info("best alpha: {:.5f}".format(best_alpha))
    alphas = mod.alphas_
    logger.debug("tested alphas:\n{}".format(alphas))
    coef = pd.Series(data=mod.coef_, index=X_columns)
    logger.debug("best coefficients:\n{}".format(coef))
#    mse_path = mod.mse_path_
    
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
    logger.info("ElasticNet regression of selected_df regarding ranking_log")
    logger.info("with alpha={:.5f} and L1_wt={}:\n{}"
                .format(best_alpha, best_l1_ratio, res))
    
    # Normality of residuals
    # Jarque-Bera test:
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(mod_sm.resid)
    logger.info("Jarque-Bera test: {}".format(lzip(name, test)))
    # Omni test:
    name = ['Chi^2', 'Two-tail probability']
    test = sms.omni_normtest(mod_sm.resid)
    logger.info("Omnibus test: {}".format(lzip(name, test)))
    
    # Multicollinearity
    # Conditional Number:
    logger.info("Conditional Number: {}"
                .format(np.linalg.cond(mod_sm.model.exog)))
    
    # Heteroskedasticity tests
    # Breush-Pagan test:
    name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
    test = sms.het_breuschpagan(mod_sm.resid, mod_sm.model.exog)
    logger.info("Breush-Pagan test: {}".format(lzip(name, test)))
    # Goldfeld-Quandt test
    name = ['F statistic', 'p-value']
    test = sms.het_goldfeldquandt(mod_sm.resid, mod_sm.model.exog)
    logger.info("Goldfeld-Quandt test: {}".format(lzip(name, test)))
    
    #%% export results as pickle file to models folder
    
    # pickle mod
    with open(os.path.join(models_path, 'sklearn_ElasticNetCV.pkl'),
              'wb') as handle:
        pickle.dump(mod, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved elastic net model of sklearn to {}."
                .format(os.path.join(models_path,
                                     'sklearn_ElasticNetCV.pkl')))
    
    # pickle mod_sm
    with open(os.path.join(models_path, 'sm_OLS_fit_regularized.pkl'),
              'wb') as handle:
        pickle.dump(mod_sm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved elastic net model of statsmodels to {}."
                .format(os.path.join(models_path,
                                     'sm_OLS_fit_regularized.pkl')))
    
    # save res as .txt
    f = open(os.path.join(models_path,
                          'sm_OLS_fit_regularized_summary.txt'), "w+")
    f.write(res)
    f.close()
    
    
    #%% logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to train Elastic Net Model: {}"
                .format(time_passed))
    
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
    args = parser.parse_args()
    
    # run main
    main(args.processed_path, args.models_path)
