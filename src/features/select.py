# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
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
    
    #%% split df into dependent and independent variables
    y, X = np.split(cleaned_df, [2], axis=1)
    start = time()
    
    #%% sets vocabulary for subsets of features
#    everything = set(X.columns)
#    loc_max = {'loc_max'}
#    radon = {x for x in X.columns if 'radon' in x}
#    radon_h = {x for x in X.columns if 'radon_h' in x}
#    radon_mi = {x for x in X.columns if 'radon_mi' in x}
#    radon_raw = {x for x in X.columns if 'radon_raw' in x}
#    radon_cc = {x for x in X.columns if '_cc' in x}
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
    
    # drop features with more than 90% zeros
    n = len(X)
    dropped = cleaned_df.columns[(cleaned_df == 0).sum() / n > .9]
    X.drop(columns=dropped, errors='ignore', inplace=True)
    logger.info(("Dropped {} features which had more than 90% zeros:\n"
                 + ('\n'+' '*56).join(dropped)).format(len(dropped)))
    # print warning if dropped important features (not in pylint_rest)
    if not all([d in pylint_rest for d in dropped]):
        logger.warning("Also dropped some higher level features: {}"
                       .format([d for d in dropped if d not in pylint_rest]))
    
    #%% multivariate feature selection
    
    # define recursive dropping function
    def drop_max_vif(X, logger, steps=-1):
        """Recursively drops feature with highest VIF, until all VIFs < 10
        or if <steps> > 0 defined: at most <steps> drops."""
        vif = pd.Series(data = [variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])],
                        index = X.columns)
        if vif.max() < 10 or steps == 0:
            return X
        else:
            drop = vif.idxmax()
            if drop not in pylint_rest:
                logger.info("Dropped {} (VIF = {}).".format(drop, vif[drop]))
            else:
                logger.debug("Dropped {} (VIF = {}).".format(drop, vif[drop]))
            return drop_max_vif(X.drop(columns=drop), logger, steps-1)
    
    # remove multi-collinearity through VIF
    logger.info("Start dropping features with high VIF.")
    n_old = X.shape[1]
    X = drop_max_vif(X, logger)
    n_new = X.shape[1]
    vif = pd.Series(data = [variance_inflation_factor(X.values, i)
                            for i in range(X.shape[1])],
                    index = X.columns)
    logger.info("Dropped {} features with VIF > 10".format(n_old-n_new))
    logger.info("Remaining {} features are:\n".format(len(vif))
                + '\n'.join([' '*56 + '{:<50} {}'.format(x, y) 
                            for (x, y) in zip(vif.index, vif)]))
    
    selected_df = pd.concat([y, X], axis=1)
    
    #%% export selected_df as pickle file to processed folder
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