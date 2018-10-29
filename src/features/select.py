# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
from time import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter


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
    
#    # optionally select subset by CompetitionId
#    counter = Counter(cleaned_df.CompetitionId)
#    cleaned_df = cleaned_df.loc[cleaned_df.CompetitionId == counter.most_common(1)[0][0]]
    
    teams_df = cleaned_df.iloc[:, :9]
    y = cleaned_df.iloc[:, 9:10]
    X = cleaned_df.iloc[:, 10:]
    n = len(X)
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
    
    #%% set list of interesting features (manually selected)
    
#    uses_module = {col for col in X.columns if 'uses_module_' in col}
    i_radon = {'radon_raw_comments_ratio',
               'radon_cc_mean',
               'radon_cc_sum_ratio',
               'radon_mi_mean',
               'radon_mi_sum_ratio',
               'radon_h_vocabulary_ratio',
               'radon_h_length_ratio',
               'radon_h_calculated_length_ratio',
               'radon_h_volume_ratio',
               'radon_h_difficulty_ratio',
               'radon_h_effort_ratio'}
    i_pylint = {'pylint_comment_ratio',
                'pylint_convention_ratio',
                'pylint_refactor_ratio',
                'pylint_warning_ratio',
                'pylint_error_ratio'}
    i_core = {'uses_module_numpy',
              'uses_module_scipy',
              'uses_module_pandas'}
    i_vis = {'uses_module_matplotlib',
             'uses_module_seaborn'}
    i_ml = {'uses_module_sklearn',
            'uses_module_statsmodels',
            'uses_module_xgboost'}
    i_dl = {'uses_module_keras',
            'uses_module_theano',
            'uses_module_pylearn2',
            'uses_module_caffe',
            'uses_module_lasagne',
            'uses_module_mxnet',
            'uses_module_nltk',
            'uses_module_gensim',
            'uses_module_pattern',
            'uses_module_nolearn'}
    
    i_uses_module = i_core | i_vis | i_ml | i_dl
    interesting = i_radon | i_pylint | i_uses_module | {'loc_max_log'}
    X.drop(columns=set(X.columns)-interesting, errors='ignore', inplace=True)
    
    counts = pd.Series(
            data = [X.shape[0] - sum(X[col] == 0)
                    for col in interesting],
            index = interesting)
    logger.info("Manually set potentially interesting features.")
    logger.debug("Interesting features sorted by counts:\n{}"
                 .format(counts.sort_values(ascending=False)))
    
    #%% aggregate uses_module features per functionality
    
    X['core'] = X.loc[:, i_core].any(axis=1).astype(int)
    X.drop(columns=i_core, inplace=True)
    X['visualization'] = X.loc[:, i_vis].any(axis=1).astype(int)
    X.drop(columns=i_vis, inplace=True)
    X['machine_learning'] = X.loc[:, i_ml].any(axis=1).astype(int)
    X.drop(columns=i_ml, inplace=True)
    X['deep_learning'] = X.loc[:, i_dl].any(axis=1).astype(int)
    X.drop(columns=i_dl, inplace=True)
    
    #%% univariate feature selection
    
    # drop features with more than 90% zeros
    dropped = X.columns[(X == 0).sum() / n > .9]
    X.drop(columns=dropped, errors='ignore', inplace=True)
    logger.info("Dropped {} features which had more than 90% zeros."
                .format(len(dropped)))
    logger.debug(("Following features had more than 90% zeros:\n"
                  + ('\n'+' '*56).join(dropped)))
    # print warning if dropped important features (not in pylint_rest)
    if not all([d in pylint_rest for d in dropped]):
        logger.warning("Dropped feature: {}"
                       .format([d for d in dropped
                                if d not in pylint_rest and
                                'uses_module' not in d]))
    
    #%% multivariate feature selection
    
#    # define recursive dropping function
#    def drop_max_vif(X, logger, steps=-1, vif_max=10):
#        """Recursively drops feature with highest VIF, until all VIFs < 10
#        or if <steps> > 0 defined: at most <steps> drops."""
#        vif = pd.Series(data = [variance_inflation_factor(X.values, i)
#                                for i in range(X.shape[1])],
#                        index = X.columns)
#        if vif.max() < vif_max or steps == 0:
#            return X
#        else:
#            drop = vif.idxmax()
#            if drop not in pylint_rest:
#                logger.warning("Dropped {} (VIF = {}).".format(drop, vif[drop]))
#            else:
#                logger.info("Dropped {} (VIF = {}).".format(drop, vif[drop]))
#            return drop_max_vif(X.drop(columns=drop), logger, steps-1, vif_max)
#    
#    # remove multi-collinearity through VIF
#    logger.info("Start dropping features with high VIF.")
#    n_old = X.shape[1]
#    X = drop_max_vif(X, logger, steps=-1, vif_max=15)
#    n_new = X.shape[1]
#    vif = pd.Series(data = [variance_inflation_factor(X.values, i)
#                            for i in range(X.shape[1])],
#                    index = X.columns)
#    logger.info("Dropped {} features with VIF > 10".format(n_old-n_new))
#    logger.info("Remaining {} features are:\n".format(len(vif))
#                + '\n'.join([' '*56 + '{:<50} {}'.format(x, y) 
#                            for (x, y) in zip(vif.index, vif)]))
    
     #%% concat teams_df, y and X to selected_df
    
    selected_df = pd.concat([teams_df, y, X], axis=1)
    
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