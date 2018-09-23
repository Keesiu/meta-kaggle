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
    Finally, drops rows with NaNs or with outliers as values, like Score > 1."""
    
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
    features_df.rename(mapper={('radon_mi', 'sum'): ('radon_mi_sum', 'sum'),
                               ('radon_mi', 'mean'): ('radon_mi_mean', 'mean')},
            axis=1, inplace=True)
    logger.info("Renamed column ('radon_mi', 'sum'): ('radon_mi_sum', 'sum').")
    logger.info("Renamed column ('radon_mi', 'mean'): ('radon_mi_mean', 'mean').")
    multi_index = pd.MultiIndex.from_tuples(features_df.columns,
                                            names=['feature', 'stat'])
    features_df.columns = multi_index
    logger.info("Created MultiIndex of features_df to split sums and means.")
    
    #%% cleaning columns
    
    # drop mean columns since mostly irrelevant
    df = features_df.drop('mean', axis=1, level=1)
    logger.info("Created df by dropping all 'mean' features. Shape of df: {}"
                .format(df.shape))
    
    # re-include relevant columns from means
    df = pd.concat([features_df[('radon_avg_cc', 'mean')],
                    features_df[('radon_mi_mean', 'mean')],
                    features_df[('radon_raw_is_error', 'mean')],
                    features_df[('radon_cc_is_error', 'mean')],
                    features_df[('radon_h_is_error', 'mean')],
                    features_df[('radon_mi_is_error', 'mean')],
                    features_df[('pylint_is_error', 'mean')],
                    df], axis=1)
    logger.info("Re-included 7 relevant 'mean' features. Shape of df: {}"
                .format(df.shape))
    
    # drop column (radon_avg_cc, sum)
    df.drop([('radon_avg_cc', 'sum')], axis=1, inplace=True)
    logger.info("Dropped column (radon_avg_cc, sum). Shape of df: {}"
                .format(df.shape))
    
    # drop halstead metrics since they are not additive thus wrong
    df.drop([('radon_h_calculated_length', 'sum'),
             ('radon_h_volume', 'sum'),
             ('radon_h_difficulty', 'sum'),
             ('radon_h_effort', 'sum'),
             ('radon_h_time', 'sum'),
             ('radon_h_bugs', 'sum')],
            axis=1, inplace=True)
    logger.info("Dropped 6 wrong calculated Halstead metrics. Shape of df: {}"
                .format(df.shape))
    
    # construct correct halstead metrics
    # see: https://radon.readthedocs.io/en/latest/intro.html#halstead-metrics
    df[('radon_h_calculated_length', 'custom')] = df[('radon_h_h1', 'sum')] \
            * np.log2(df[('radon_h_h1', 'sum')]) \
            + df[('radon_h_h2', 'sum')] \
            * np.log2(df[('radon_h_h2', 'sum')])
    df[('radon_h_volume', 'custom')] = df[('radon_h_length', 'sum')] \
            * np.log2(df[('radon_h_vocabulary', 'sum')])
    df[('radon_h_difficulty', 'custom')] = df[('radon_h_h1', 'sum')] / 2 \
            * df[('radon_h_N2', 'sum')] / df[('radon_h_h2', 'sum')]
    df[('radon_h_effort', 'custom')] = df[('radon_h_difficulty', 'custom')] \
            * df[('radon_h_volume', 'custom')]
    df[('radon_h_time', 'custom')] = df[('radon_h_effort', 'custom')] / 18
    df[('radon_h_bugs', 'custom')] = df[('radon_h_volume', 'custom')] / 3000
    logger.info("Constructed 6 Halstead metrics correctly. Shape of df: {}"
                .format(df.shape))
    
    #%% cleaning rows
    
    # delete repos with to many errors during feature extraction
    df = df[df[('radon_raw_is_error', 'mean')] != 1]
    df = df[df[('radon_cc_is_error', 'mean')] != 1]
    df = df[df[('radon_h_is_error', 'mean')] != 1]
    df = df[df[('radon_mi_is_error', 'mean')] != 1]
    df = df[df[('pylint_is_error', 'mean')] != 1]
    logger.info("Dropped all rows with error rates = 1. Shape of df: {}"
                .format(df.shape))
    
    # delete rows of error rates
    df.drop(columns=[('radon_raw_is_error', 'mean'),
                     ('radon_cc_is_error', 'mean'),
                     ('radon_h_is_error', 'mean'),
                     ('radon_mi_is_error', 'mean'),
                     ('pylint_is_error', 'mean'),
                     ('radon_raw_is_error', 'sum'),
                     ('radon_cc_is_error', 'sum'),
                     ('radon_h_is_error', 'sum'),
                     ('radon_mi_is_error', 'sum'),
                     ('pylint_is_error', 'sum'),], inplace=True)
    logger.info("Dropped 10 is_error columns. Shape of df: {}"
                .format(df.shape))
    
    # delete rows with NaNs
    df.dropna(inplace=True)
    logger.info("Dropped all rows with NaNs. Shape of df: {}"
                .format(df.shape))
    
    # drop second level of MultiIndex
    df.columns = df.columns.droplevel(level=1)
    
    # concatenate df with Score and Ranking
    df = pd.concat([pd.to_numeric(teams_df.Score),
                    pd.to_numeric(teams_df.Ranking),
                    df],
                    join='inner', axis=1)
    logger.info("Concatenated df with Score and Ranking. Shape of df: {}"
                .format(df.shape))
    
    # drop repos with outlying scores
    df = df[df.Score <= 1]
    logger.info("Dropped rows with Scores > 1. Shape of df: {}"
                .format(df.shape))
    
    #%% export df as pickle file to processed folder
    aggregated_df.to_pickle(os.path.join(processed_path, 'df.pkl'))
    logger.info("Saved repos_df to {}."
            .format(os.path.join(processed_path, 'df.pkl')))
    
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
            '-i', '--interim_path',
            default = "data/interim",
            help = """path to aggregated features aggregated_df.pkl
                    (default: data/interim)""")
    parser.add_argument(
            '-p', '--processed_path',
            default = "data/processed",
            help = """path to store the cleaned output df.pkl
                    (default: data/processed)""")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path, args.processed_path)