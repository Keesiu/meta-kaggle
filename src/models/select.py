# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression, SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main(processed_path = "data/processed"):
    
    """Selects features for training the model."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    
    # load cleaned_df
    cleaned_df = pd.read_pickle(os.path.join(processed_path, 'cleaned_df.pkl'))
    logger.info("Loaded cleaned_df.pkl. Shape of df: {}"
                .format(cleaned_df.shape))
    
    #%% pre-work
    
    # split df into dependent and independent variables
    y, X = np.split(cleaned_df, [2], axis=1)
    
    # sets vocabulary for subsets of features
    everything = set(X.columns)
    loc_max = {'loc_max'}
    radon = {x for x in X.columns if 'radon' in x}
    radon_h = {x for x in X.columns if 'radon_h' in x}
    radon_mi = {x for x in X.columns if 'radon_mi' in x}
    radon_raw = {x for x in X.columns if 'radon_raw' in x}
    radon_cc = {x for x in X.columns if '_cc' in x}
    pylint = {x for x in X.columns if 'pylint' in x}
    pylint_raw = {'pylint_code_ratio',
                  'pylint_docstring_ratio',
                  'pylint_comment_ratio',
                  'pylint_empty_ratio'}
    pylint_dup = {'pylint_nb_duplicated_lines_ratio'}
    pylint_cat = {'pylint_convention_ratio',
                  'pylint_refactor_ratio',
                  'pylint_warning_ratio',
                  'pylint_error_ratio'}
    pylint_rest = pylint - pylint_raw - pylint_dup - pylint_cat
    
    #%% univariate feature selection
    start = time()
    
    # drop features with more than 50% zeros
    n = len(X)
    dropped = set()
    col_old = set(X.columns)
    drop_now = cleaned_df.columns[(cleaned_df == 0).sum() / n > .5]
    X.drop(columns=drop_now, errors='ignore', inplace=True)
    dropped.update(drop_now)
    col_new = set(X.columns)
    for drop in col_old-col_new:
        logger.debug("Dropped: {}".format(drop))
    logger.info("{} features have more than 50% zeros. Dropped {}."
                .format(len(drop_now), len(col_old)-len(col_new)))
    if not all([d in pylint_rest for d in dropped]):
        logger.warning("Dropped some higher level features: {}"
                       .format([d for d in dropped if d not in pylint_rest]))
    #%%
    # compute Pearson correlation coefficient and
    # p-value for testing non-correlation (not reliable for small datasets)
    corr_score = pd.DataFrame(
            data = [list(pearsonr(X[x], y.score)) for x in X],
            index = X.columns,
            columns = ['corr_coeff', 'p_value'])
    corr_ranking_log = pd.DataFrame(
            data = [list(pearsonr(X[x], y.ranking_log)) for x in X],
            index = X.columns,
            columns = ['corr_coeff', 'p_value'])
    
    # compute f_scores
    f_score, pval = f_regression(X, y.score)    
    f_score_df = pd.DataFrame(
            data = {'f_score' : f_score,
                    'pval' : pval},
            index = X.columns)
    f_score, pval = f_regression(X, y.ranking_log)
    f_ranking_log_df = pd.DataFrame(
            data = {'f_score' : f_score,
                    'pval' : pval},
            index = X.columns)
    del f_score, pval
    
    selector = SelectKBest(f_regression, k=10)
    selector.fit(X, y.score)
    mask = selector.get_support()
    X = X.loc[:, mask]
    

    # coefficient of variation (ratio of biased standard deviation to mean)
    X.std()/X.mean()
    
    # dropping features manually by domain knowledge
    mask = everything - pylint_rest - pylint_dup - {'radon_h_h1_ratio',
                                                    'radon_h_h2_ratio',
                                                    'radon_h_N1_ratio',
                                                    'radon_h_N2_ratio',
                                                    'loc_max',
                                                    'radon_h_effort_ratio'}
    X = X.loc[:, list(mask)]

    #%% multivariate feature selection
    
    # remove multi-collinearity through VIF
    def drop_max_vif(X, logger, steps=-1):
        vif = pd.Series(data = [variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])],
                        index = X.columns)
        if vif.max() < 5 or steps == 0:
            return X
        else:
            drop = vif.idxmax()
            logger.warning("Dropped {} (VIF = {}).".format(drop, vif[drop]))
            return drop_max_vif(X.drop(columns=drop), logger, steps-1)
    X = drop_max_vif(X, logger)
    vif = pd.Series(data = [variance_inflation_factor(X.values, i)
                            for i in range(X.shape[1])],
                    index = X.columns)
    
    #%% export selected_df as pickle file to processed folder
    selected_df = pd.concat([y, X], axis=1)
    selected_df.to_pickle(os.path.join(processed_path, 'selected_df.pkl'))
    logger.info("Saved selected_df to {}."
            .format(os.path.join(processed_path, 'selected_df.pkl')))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to select the features: {}".format(time_passed))

#%%
if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/select.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Selects features for training the model.")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to load the cleaned data df.pkl \
                    (default: data/processed)")
    args = parser.parse_args()
    
    # run main
    main(args.processed_path)