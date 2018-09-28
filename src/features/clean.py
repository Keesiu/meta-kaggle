# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time


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
    aggregated_df = pd.read_pickle(os.path.join(interim_path, 'aggregated_df.pkl'))
    logger.info("Loaded aggregated_df.pkl. Shape of aggregated_df: {}"
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
    cleaned_df = features_df.drop('mean', axis=1, level=1)
    logger.info("Created cleaned_df by dropping all 'mean' features. Shape: {}"
                .format(cleaned_df.shape))
    
    # re-include relevant columns from means
    cleaned_df = pd.concat([features_df[('radon_avg_cc', 'mean')],
                            features_df[('radon_mi_mean', 'mean')],
                            features_df[('radon_raw_is_error', 'mean')],
                            features_df[('radon_cc_is_error', 'mean')],
                            features_df[('radon_h_is_error', 'mean')],
                            features_df[('radon_mi_is_error', 'mean')],
                            features_df[('pylint_is_error', 'mean')],
                            cleaned_df], axis=1)
    logger.info("Re-included 7 relevant 'mean' features. Shape: {}"
                .format(cleaned_df.shape))
    
    # drop column (content_len, sum) since it's irrelevant
    cleaned_df.drop([('content_len', 'sum')], axis=1, inplace=True)
    logger.info("Dropped column (content_len, sum). Shape: {}"
                .format(cleaned_df.shape))
    
    # drop column (radon_avg_cc, sum) since it doesn't make sense
    cleaned_df.drop([('radon_avg_cc', 'sum')], axis=1, inplace=True)
    logger.info("Dropped column (radon_avg_cc, sum). Shape: {}"
                .format(cleaned_df.shape))
    
    # drop halstead metrics since they are not additive thus wrong
    cleaned_df.drop([('radon_h_calculated_length', 'sum'),
                     ('radon_h_volume', 'sum'),
                     ('radon_h_difficulty', 'sum'),
                     ('radon_h_effort', 'sum'),
                     ('radon_h_time', 'sum'),
                     ('radon_h_bugs', 'sum')],
                    axis=1, inplace=True)
    logger.info("Dropped 6 wrong calculated Halstead metrics. Shape: {}"
                .format(cleaned_df.shape))
    
    # construct correct halstead metrics
    # see: https://radon.readthedocs.io/en/latest/intro.html#halstead-metrics
    cleaned_df[('radon_h_calculated_length', 'custom')] = cleaned_df[('radon_h_h1', 'sum')] \
            * np.log2(cleaned_df[('radon_h_h1', 'sum')]) \
            + cleaned_df[('radon_h_h2', 'sum')] \
            * np.log2(cleaned_df[('radon_h_h2', 'sum')])
    cleaned_df[('radon_h_volume', 'custom')] = cleaned_df[('radon_h_length', 'sum')] \
            * np.log2(cleaned_df[('radon_h_vocabulary', 'sum')])
    cleaned_df[('radon_h_difficulty', 'custom')] = cleaned_df[('radon_h_h1', 'sum')] / 2 \
            * cleaned_df[('radon_h_N2', 'sum')] / cleaned_df[('radon_h_h2', 'sum')]
    cleaned_df[('radon_h_effort', 'custom')] = cleaned_df[('radon_h_difficulty', 'custom')] \
            * cleaned_df[('radon_h_volume', 'custom')]
    # Following two Halstead metrics are skipped since propotional to other
    # cleaned_df[('radon_h_time', 'custom')] = cleaned_df[('radon_h_effort', 'custom')] / 18
    # cleaned_df[('radon_h_bugs', 'custom')] = cleaned_df[('radon_h_volume', 'custom')] / 3000
    logger.info("Re-constructed 4 relevant Halstead metrics. Shape: {}"
                .format(cleaned_df.shape))
    
    # transforms features to become ratios of loc to avoid multi-collinearity
    # calculate maximum number for lines of codes, since inconsistent
    max_loc_df = pd.concat([
            cleaned_df[('radon_raw_loc', 'sum')],
            cleaned_df[('radon_raw_lloc', 'sum')],
            cleaned_df[('radon_raw_sloc', 'sum')]
                    + cleaned_df[('radon_raw_multi', 'sum')]
                    + cleaned_df[('radon_raw_single_comments', 'sum')]
                    + cleaned_df[('radon_raw_blank', 'sum')],
            cleaned_df[('pylint_code', 'sum')]
                    + cleaned_df[('pylint_docstring', 'sum')]
                    + cleaned_df[('pylint_comment', 'sum')]
                    + cleaned_df[('pylint_empty', 'sum')]], axis=1)
    cleaned_df[('loc_max', 'max')] = max_loc_df.max(axis=1)
    logger.info("Created 'loc_max'. Shape: {}"
                .format(cleaned_df.shape))
    cleaned_df.drop(columns=[('radon_raw_loc', 'sum'), ('radon_raw_lloc', 'sum')],
            inplace=True)
    logger.info("Dropped 'radon_raw_loc' and 'radon_raw_lloc'. Shape: {}"
                .format(cleaned_df.shape))
    # create list of columns, which are correlated/dependent to max_loc
    dependent = []
    for col in cleaned_df.columns:
        if col[1] == 'mean' or '_is_error' in col[0] or col[0] == 'loc_max':
            logger.debug("Skipped independent feature: {}".format(col))
            continue
        elif 'pylint' in col[0] or 'radon' in col[0]:
            dependent.append(col)
            logger.debug("Included dependent feature:  {}".format(col))
        else:
            logger.error("{} not catched during creation of list 'dependent'."
                         .format(col))
    for col in dependent:
        cleaned_df[(col[0]+'_ratio', 'ratio')] = cleaned_df[col]/cleaned_df[('loc_max', 'max')]
        logger.debug("Created new ratio feature for: {}".format(col))
    cleaned_df.drop(columns=dependent, inplace=True)
    logger.info("Transformed {} features to ratios of lines-of-code. Shape: {}"
                .format(len(dependent), cleaned_df.shape))
    
    #%% cleaning rows
    
    # delete repos with too many errors during feature extraction
    cleaned_df = cleaned_df[cleaned_df[('radon_raw_is_error', 'mean')] != 1]
    cleaned_df = cleaned_df[cleaned_df[('radon_cc_is_error', 'mean')] != 1]
    cleaned_df = cleaned_df[cleaned_df[('radon_h_is_error', 'mean')] != 1]
    cleaned_df = cleaned_df[cleaned_df[('radon_mi_is_error', 'mean')] != 1]
    cleaned_df = cleaned_df[cleaned_df[('pylint_is_error', 'mean')] != 1]
    logger.info("Dropped all rows with error rates = 1. Shape: {}"
                .format(cleaned_df.shape))
    
    # delete rows of error rates
    cleaned_df.drop(columns=[('radon_raw_is_error', 'mean'),
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
                .format(cleaned_df.shape))
    
    # drop second level of MultiIndex
    cleaned_df.columns = cleaned_df.columns.droplevel(level=1)    
    
    # concatenate cleaned_df with Score and Ranking
    cleaned_df = pd.concat([pd.to_numeric(teams_df.Score),
                    pd.to_numeric(teams_df.Ranking),
                    cleaned_df],
                    join='inner', axis=1)
    logger.info("Concatenated cleaned_df with Score and Ranking. Shape: {}"
                .format(cleaned_df.shape))
    
    # drop repos with outlying scores
    cleaned_df = cleaned_df[cleaned_df.Score <= 1]
    logger.info("Dropped rows with Scores > 1. Shape: {}"
                .format(cleaned_df.shape))
    
    # drop repos with outlying length
    cleaned_df = cleaned_df[cleaned_df.loc_max <= 20000]
    logger.info("Dropped rows with lines of code > 20.000. Shape: {}"
                .format(cleaned_df.shape))
    
    # delete rows with NaNs
    cleaned_df.dropna(inplace=True)
    logger.info("Dropped all rows with NaNs. Shape: {}"
                .format(cleaned_df.shape))
    
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