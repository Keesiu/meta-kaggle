# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_validate
import statsmodels.api as sm
import pickle


def main(processed_path = "data/processed",
         models_path = "models",
         df_name = 'cleaned',
         y_name = 'ranking_log'):
    
    """Trains the model."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    models_path = os.path.normpath(models_path)
    logger.debug("Path to models normalized: {}"
                 .format(models_path))
    
    # load df, either cleaned, cleaned_pca, selected, selected_pca
    df = pd.read_pickle(os.path.join(processed_path, df_name+'_df.pkl'))
    logger.info("Loaded '{}'. Shape of df: {}"
                .format(df_name+'_df.pkl', df.shape))
    # wheather PCA was performed
    PCA = True if df_name[-4:] == '_pca' else False
    
    # split df into dependent and independent variables
    y, X = np.split(df, [2], axis=1)
    X_columns = X.columns
    X_index = X.index
    X = X.values
    # set y to either ranking_log or score_neg_log
    y = y[y_name].values
    logger.info("Set y to '{}'.".format(y_name))
    
    #%% Nested 10-fold cross-validation for linear regression of ranking_log
    #   with lasso regularization (inner CV for alpha tuning, outer for R^2 robustness)
    
    start = time()
    
    # define hyperparameter
#    # define list of 100 alphas to test: from 1 logarithmically decreasing to 0
#    BASE = 1 + 1/5
#    logger.debug("Constant BASE is set to {}.".format(BASE))
#    ALPHAS = [BASE**(-x) for x in range(100)]
    ALPHAS = [.00001, .0001, .001, .01, .1, .25, .5, .75, .95, .99999]
    logger.debug("Alphas set to {}".format(ALPHAS))
    # define list of l1-ratios to test
    # Note that a good choice of list of values for l1_ratio is often to put
    # more values close to 1 (i.e. Lasso) and less close to 0 (i.e. Ridge),
    # as in [.1, .5, .7, .9, .95, .99, 1]
    L1_RATIOS = [1-a for a in ALPHAS]
    logger.debug("L1_ratios set to {}".format(L1_RATIOS))
    # define other hyperparameter
    CV = 10
    RS = 42
    N_JOBS = -1
    SELECTION = 'random'
    # normalize data only if PCA was not performed (because PCA standardized)
    # If True, the regressors X will be normalized before regression
    # by subtracting the mean and dividing by the l2-norm.
    NORMALIZE = not PCA
    logger.debug("CV={}, RS={}, N_JOBS={}, SELECTION={}, NORMALIZE={}"
                 .format(CV, RS, N_JOBS, SELECTION, NORMALIZE))
    
    # print R^2 values for bounding alphas 0 and 1 to make sense of alphas
    logger.info("R^2 for alpha=0: {}"
                .format(ElasticNet(alpha=0, l1_ratio=.5, normalize=NORMALIZE, random_state=42)
                        .fit(X, y)
                        .score(X, y)))
    logger.info("R^2 for alpha=1: {}"
                .format(ElasticNet(alpha=1, l1_ratio=.5, normalize=NORMALIZE, random_state=42)
                        .fit(X, y)
                        .score(X, y)))
    
    # train model
    mod = ElasticNetCV(cv=CV, alphas=ALPHAS, l1_ratio=L1_RATIOS, normalize=NORMALIZE,
                       random_state=RS, selection=SELECTION, n_jobs= N_JOBS) \
          .fit(X, y)
    
    # log some statistics
    logger.info("best R^2 score: {}".format(mod.score(X, y)))
    l1_ratio = mod.l1_ratio_
    logger.info("best l1_ratio: {}".format(l1_ratio))
    alpha = mod.alpha_
    logger.info("best alpha: {}".format(alpha))
    coef = pd.Series(data=mod.coef_, index=X_columns)
    logger.debug("best coefficients:\n{}".format(coef))
    
    # Nested Cross-Validation to test robustness of R^2
    cv_results = cross_validate(ElasticNetCV(cv=CV,
                                             alphas=ALPHAS,
                                             normalize=NORMALIZE,
                                             random_state=RS,
                                             selection=SELECTION,
                                             n_jobs= N_JOBS),
                                X, y, cv=10,
                                return_train_score=True, n_jobs=N_JOBS)
    logger.info("95% confidence intervall: {:05.2f} +/-{:05.2f} (mean +/- 2 std)"
                .format(cv_results['test_score'].mean(),
                        cv_results['test_score'].std()*2))
    logger.debug("Nested cross-validation results:\n{}"
                .format(pd.DataFrame(data=cv_results)))
    
    # Elastic Net regression with statsmodels for summary
    mod2 = sm.OLS(y, sm.add_constant(pd.DataFrame(data=X,
                                                  columns=X_columns,
                                                  index=X_index)))\
          .fit_regularized(method='elastic_net',
                           alpha=alpha,
                           L1_wt=l1_ratio,
                           refit=True)
    res = mod2.summary().as_text()
    logger.info("ElasticNet regression (from sm) of '{}' with respect to '{}' with alpha={} and L1_wt={}:\n{}"
                .format(df_name, y_name, alpha, l1_ratio, res))
    
    #%% export results as pickle file to models folder
    
    # pickle mod
    with open(os.path.join(models_path, df_name+'_'+y_name+'_sklearn_ElasticNetCV.pkl'), 'wb') as handle:
        pickle.dump(mod, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved elastic net model of sklearn to {}."
                .format(os.path.join(models_path, df_name+'_'+y_name+'_sklearn_ElasticNetCV.pkl')))
    
    # pickle mod2
    with open(os.path.join(models_path, df_name+'_'+y_name+'_sm_OLS_fit_regularized.pkl'), 'wb') as handle:
        pickle.dump(mod2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved elastic net model of statsmodels to {}."
                .format(os.path.join(models_path, df_name+'_'+y_name+'_sm_OLS_fit_regularized.pkl')))
    
    # save res as .txt
    f = open(os.path.join(models_path, df_name+'_'+y_name+'_sm_OLS_fit_regularized_summary.txt'), "w+")
    f.write(res)
    f.close()
    
    
    #%% logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to train Elastic Net Model on '{}' with respect to '{}': {}"
                .format(df_name+'_df.pkl', y_name, time_passed))
    
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
            '--df_name',
            default = 'cleaned',
            help = "name of df to be trained on \
                    either cleaned, cleaned_pca, selected, selected_pca \
                    (default: cleaned)")
    parser.add_argument(
            '--y_name',
            default = 'ranking_log',
            help = "name of dependent variable to be trained on \
                    either 'ranking_log' or 'score_neg_log' \
                    (default: ranking_log)")
    args = parser.parse_args()
    
    # run main
    main(args.processed_path, args.models_path, args.df_name, args.y_name)
