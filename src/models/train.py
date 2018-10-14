# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


def main(processed_path = "data/processed",
         models_path = "models",
         df_name = 'cleaned'):
    
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
    logger.info("Loaded {}. Shape of df: {}"
                .format(df_name+'_df.pkl', df.shape))
    
    # split df into dependent and independent variables
    y, X = np.split(df, [2], axis=1)
    X_columns = X.columns
    X_index = X.index
    y_s = y.score
    y_r = y.ranking_log
    
    #%% start training
    
    
#    # train-test-split
#    X_train, X_test, y_train, y_test  = train_test_split(
#            X, y, test_size=0.3, random_state=42)
    
#    # Linear regression with sklearn
#    lr = LinearRegression()
#    lr.fit(X_train, y_train)
#    lr.score(X_test, y_test)
#    lr.get_params()
#    lr.coef_
#    plt.plot(X, lr.predict(X))
#    plt.show()
    
#    # Linear regression with statsmodels
#    mod = sm.OLS(y.ranking_log, sm.add_constant(X))
#    res = mod.fit()
#    print(res.summary())
    
#    # logistic regression with statsmodels
#    mod = sm.Logit(y.score, sm.add_constant(X))
#    res = mod.fit()
#    print(res.summary())
    
#    # lasso logistic regression with statsmodels
#    mod = sm.Logit(y.score, sm.add_constant(X))
#    res = mod.fit_regularized(method='l1', alpha=.62)
#    print(res.summary())
    
#    # LR with statsmodels in R-style
#    model = ols("Ranking ~ radon_sum_cc_ratio + pylint_class_ratio", selected_df)
#    results = model.fit()
#    results.summary()
    
    #%% Nested 10-fold cross-validation for linear regression of ranking_log
    #   with lasso regularization (inner CV for alpha tuning, outer for R^2 robustness)
    
    start = time()
    
    # print R^2 values for bounding alphas 0 and 1 to make sense of alphas
    logger.info("R^2 for alpha = 0: {}"
                .format(Lasso(alpha=0, random_state=42)
                        .fit(X.values, y_r.values)
                        .score(X.values, y_r.values)))
    logger.info("R^2 for alpha = 1: {}"
                .format(Lasso(alpha=1, random_state=42)
                        .fit(X.values, y_r.values)
                        .score(X.values, y_r.values)))
    
    # define list of 100 alphas to test: from 1 logarithmically decreasing to 0
    BASE = 1 + 1/5
    logger.debug("Constant BASE is set to {}.".format(BASE))
    alphas = [BASE**(-x) for x in range(100)]
    
    # define hyperparameter
    CV = 20
    RS = 42
    N_JOBS = -1
    SELECTION = 'random'
    
    # train LassoCV
    t1 = time()
    lasso = LassoCV(cv=CV, alphas=alphas,
                    random_state=RS, selection=SELECTION, n_jobs= N_JOBS) \
            .fit(X.values, y_r.values)
    t2 = time()
    print("Time to perform cross-validation: {}"
          .format(pd.Timedelta(seconds=t2-t1).round(freq='s')))
    
    # log some statistics
    logger.info("Lasso score = {}.".format(lasso.score(X.values, y_r.values)))
    lasso_alpha = lasso.alpha_
    logger.info("Lasso alpha = {}.".format(lasso_alpha))
    lasso_coef = pd.Series(data=lasso.coef_, index=X_columns)
    logger.info("Lasso coefficients = {}.".format(lasso_coef))
    lasso_mse_path = pd.DataFrame(data=lasso.mse_path_, index=alphas)
    logger.info("Lasso MSE = {}.".format(lasso_mse_path))
    
    # plot all folds with all
    
    
    m_log_alphas = -np.log(lasso.alphas_)/np.log(BASE)
    # Display results
    plt.figure(figsize=(10,8))
    ymin, ymax = 0, 1000
    plt.plot(m_log_alphas, lasso.mse_path_, ':')
    plt.plot(m_log_alphas, lasso.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(lasso.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    
    plt.legend()
    
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold')
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    plt.show()
    
    # Nested Cross-Validation
    lasso = LassoCV(cv=CV, alphas=alphas,
                    random_state=RS, selection=SELECTION, n_jobs= N_JOBS)
    t1 = time()
    cv_results = cross_validate(lasso, X.values, y_r.values,
                                return_train_score=True, cv=10, n_jobs=N_JOBS)
    t2 = time()
    print("Time to perform nested cross-validation: {}"
          .format(pd.Timedelta(seconds=t2-t1).round(freq='s')))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    # Linear regression with lasso regularization
    mod = sm.OLS(y_r, sm.add_constant(X))
    res = mod.fit_regularized(method='elastic_net', alpha=lasso_alpha, L1_wt=1.0, refit=True)
    print(res.summary())
    
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
