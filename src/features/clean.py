# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.utils import shuffle


def main(interim_path = "data/interim",
         processed_path = "data/processed"):
    
    """Cleans aggregated data and saves it to <processed_path>.
    
    Drops unnecessary columns, and construct new, interesting ones.
    Also drops rows, which has to high error rates during feature extraction.
    Then drops rows with NaNs or with outliers as values, like Score > 1."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    interim_path = os.path.normpath(interim_path)
    logger.debug("Path to interim data normalized: {}"
                 .format(interim_path))
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    
    # load aggregated_df
    aggregated_df = pd.read_pickle(os.path.join(interim_path,
                                                'aggregated_df.pkl'))
    logger.info("Loaded aggregated_df.pkl. Shape of df: {}"
                .format(aggregated_df.shape))
    
    #%% start cleaning
    start = time()
    
    # split aggregated_df
    teams_df, features_df = aggregated_df.iloc[:,:9], aggregated_df.iloc[:,9:]
    logger.info("agrgregated_df splitted to teams_df {} & features_df {}."
                .format(teams_df.shape, features_df.shape))
    
    # create multi-index
    # make features names unique for which both sum and mean are kept
    features_df.rename(mapper={('radon_mi', 'sum'):('radon_mi_sum', 'sum'),
                               ('radon_mi', 'mean'):('radon_mi_mean', 'mean')},
            axis=1, inplace=True)
    logger.info("Renamed col ('radon_mi', 'sum'): ('radon_mi_sum', 'sum').")
    logger.info("Renamed col ('radon_mi', 'mean'): ('radon_mi_mean', 'mean').")
    multi_index = pd.MultiIndex.from_tuples(features_df.columns,
                                            names=['feature', 'stat'])
    features_df.columns = multi_index
    logger.info("Created MultiIndex of features_df to split sums and means.")
    
    #%% cleaning columns
    
    # drop mean columns since mostly irrelevant
    X = features_df.drop('mean', axis=1, level=1)
    logger.info("Created X by dropping all 'mean' features. Shape: {}"
                .format(X.shape))
    
    # re-include relevant columns from means
    X = pd.concat([features_df[('radon_cc_mean', 'mean')],
                            features_df[('radon_mi_mean', 'mean')],
                            features_df[('radon_raw_is_error', 'mean')],
                            features_df[('radon_cc_is_error', 'mean')],
                            features_df[('radon_h_is_error', 'mean')],
                            features_df[('radon_mi_is_error', 'mean')],
                            features_df[('pylint_is_error', 'mean')],
                            X], axis=1)
    logger.info("Re-included 7 relevant 'mean' features. Shape: {}"
                .format(X.shape))
    
    # drop column (radon_cc_mean, sum) since it doesn't make sense
    X.drop([('radon_cc_mean', 'sum')], axis=1, inplace=True)
    logger.info("Dropped column (radon_cc_mean, sum). Shape: {}"
                .format(X.shape))
    
    # drop halstead metrics since they are not additive thus wrong
    X.drop([('radon_h_calculated_length', 'sum'),
            ('radon_h_volume', 'sum'),
            ('radon_h_difficulty', 'sum'),
            ('radon_h_effort', 'sum'),
            ('radon_h_time', 'sum'),
            ('radon_h_bugs', 'sum')],
            axis=1, inplace=True)
    logger.info("Dropped 6 wrong calculated Halstead metrics. Shape: {}"
                .format(X.shape))
    
    # construct correct halstead metrics
    # see: https://radon.readthedocs.io/en/latest/intro.html#halstead-metrics
    X[('radon_h_calculated_length', 'custom')] = X[('radon_h_h1', 'sum')] \
            * np.log2(X[('radon_h_h1', 'sum')]) \
            + X[('radon_h_h2', 'sum')] \
            * np.log2(X[('radon_h_h2', 'sum')])
    X[('radon_h_volume', 'custom')] = X[('radon_h_length', 'sum')] \
            * np.log2(X[('radon_h_vocabulary', 'sum')])
    X[('radon_h_difficulty', 'custom')] = X[('radon_h_h1', 'sum')] / 2 \
            * X[('radon_h_N2', 'sum')] / X[('radon_h_h2', 'sum')]
    X[('radon_h_effort', 'custom')] = X[('radon_h_difficulty', 'custom')] \
            * X[('radon_h_volume', 'custom')]
    # Following two Halstead metrics are skipped since propotional to other
    # X[('radon_h_time', 'custom')] = X[('radon_h_effort', 'custom')] / 18
    # X[('radon_h_bugs', 'custom')] = X[('radon_h_volume', 'custom')] / 3000
    logger.info("Re-constructed 4 relevant Halstead metrics. Shape: {}"
                .format(X.shape))
    
    # transforms features to become ratios of loc to avoid multi-collinearity
    # calculate maximum number for lines of codes, since inconsistent
    loc_max = pd.concat([
            X[('radon_raw_loc', 'sum')],
            X[('radon_raw_lloc', 'sum')],
            X[('radon_raw_sloc', 'sum')]
                    + X[('radon_raw_multi', 'sum')]
                    + X[('radon_raw_single_comments', 'sum')]
                    + X[('radon_raw_blank', 'sum')],
            X[('pylint_code', 'sum')]
                    + X[('pylint_docstring', 'sum')]
                    + X[('pylint_comment', 'sum')]
                    + X[('pylint_empty', 'sum')]], axis=1)
    loc_max = loc_max.max(axis=1)
    X.drop(columns=[('radon_raw_loc', 'sum'), ('radon_raw_lloc', 'sum')],
            inplace=True)
    logger.info("Dropped 'radon_raw_loc' and 'radon_raw_lloc'. Shape: {}"
                .format(X.shape))
    # create list of columns, which are correlated/dependent to max_loc
    dependent = []
    for col in X.columns:
        if col[1] == 'mean' or '_is_error' in col[0] or col[0] == 'loc_max':
            logger.debug("Skipped independent feature: {}".format(col))
            continue
        elif 'pylint' in col[0] or 'radon' in col[0]:
            dependent.append(col)
            logger.debug("Included dependent feature:  {}".format(col))
        elif 'uses_module_' in col[0]:
            # set every value > 0 to 1 because you only want indication
            X[col].where(X[col]==0, other=1, inplace=True)
            logger.debug("Skipped independent feature: {}".format(col))
            continue
        else:
            logger.error("{} not catched during creation of list 'dependent'."
                         .format(col))
    for col in dependent:
        X[(col[0]+'_ratio', 'ratio')] = X[col]/loc_max
        logger.debug("Created new ratio feature for: {}".format(col))
    X.drop(columns=dependent, inplace=True)
    logger.info("Transformed {} features to ratios of lines-of-code. Shape: {}"
                .format(len(dependent), X.shape))
    # create new column as log-transformed loc_max
    X[('loc_max_log', 'max')] = np.log(loc_max)
    logger.info("Added column 'loc_max_log'. Shape: {}"
                .format(X.shape))
    
    #%% cleaning rows
    
    # delete repos with too many errors during feature extraction
    X = X[X[('radon_raw_is_error', 'mean')] != 1]
    X = X[X[('radon_cc_is_error', 'mean')] != 1]
    X = X[X[('radon_h_is_error', 'mean')] != 1]
    X = X[X[('radon_mi_is_error', 'mean')] != 1]
    X = X[X[('pylint_is_error', 'mean')] != 1]
    logger.info("Dropped all rows with error rates = 1. Shape: {}"
                .format(X.shape))
    
    # drop columns of error rates
    X.drop(columns=[('radon_raw_is_error', 'mean'),
                     ('radon_cc_is_error', 'mean'),
                     ('radon_h_is_error', 'mean'),
                     ('radon_mi_is_error', 'mean'),
                     ('pylint_is_error', 'mean'),
                     ('radon_raw_is_error', 'sum'),
                     ('radon_cc_is_error', 'sum'),
                     ('radon_h_is_error', 'sum'),
                     ('radon_mi_is_error', 'sum'),
                     ('pylint_is_error', 'sum'),], inplace=True)
    logger.info("Dropped 10 is_error columns. Shape: {}"
                .format(X.shape))
    
    # drop second level of MultiIndex
    X.columns = X.columns.droplevel(level=1)    
    
    # concatenate X with log-transfomated Ranking
    X = pd.concat(
            [np.log(pd.to_numeric(teams_df.Ranking)).rename('ranking_log'), X],
            join='inner', axis=1)
    logger.info("Concatenated log-transformed 'ranking_log'. Shape: {}"
                .format(X.shape))
    
    # drop repos with outliers
#    desc = X.describe().T
#    top = (desc['max'] - desc['50%']) / desc['std']
#    bottom = (desc['50%']-desc['min']) / desc['std']
#    desc.loc[top>5]
#    desc.loc[bottom>3]
#    X = X[X.score <= 1] # 16
#    logger.info("Dropped rows with score > 1. Shape: {}"
#                .format(X.shape))
    X = X[(4 <= X.loc_max_log) & (X.loc_max_log <= 10)] # 2
    logger.info("Dropped rows with lines of code > 10.000. Shape: {}"
                .format(X.shape))
    X = X[X.pylint_warning_ratio <= 1] # 3
    logger.info("Dropped rows with pylint_warning_ratio > 1. Shape: {}"
                .format(X.shape))
    X = X[X.radon_h_effort_ratio <= 1000] # 6
    logger.info("Dropped rows with radon_h_effort_ratio > 1000. Shape: {}"
                .format(X.shape))
    X = X[X.radon_h_difficulty_ratio <= .07] # 2
    logger.info("Dropped rows with radon_h_difficulty_ratio > 0.1. Shape: {}"
                .format(X.shape))
    X = X[X.radon_cc_mean <= 9] # 4
    logger.info("Dropped rows with radon_cc_mean > 7.5. Shape: {}"
                .format(X.shape))
    
    # delete rows with NaNs
    X.dropna(inplace=True)
    logger.info("Dropped all rows with NaNs. Shape: {}"
                .format(X.shape))
    
    #%% concat with teams_df shuffle the data
    cleaned_df = pd.concat([teams_df, X], join='inner', axis=1)
    cleaned_df = shuffle(cleaned_df, random_state=0)
    
    #%% export cleaned_df as pickle file to processed folder
    cleaned_df.to_pickle(os.path.join(processed_path, 'cleaned_df.pkl'))
    logger.info("Saved cleaned_df to {}."
            .format(os.path.join(processed_path, 'cleaned_df.pkl')))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to clean the features: {}".format(time_passed))

#%%
if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/clean.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Cleans aggregated data.")
    parser.add_argument(
            '--interim_path',
            default = "data/interim",
            help = "path to aggregated features aggregated_df.pkl \
                    (default: data/interim)")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to store the cleaned output cleaned_df.pkl \
                    (default: data/processed)")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path, args.processed_path)