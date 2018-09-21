# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns


def main(interim_path = "data/interim",
         processed_path = "data/processed"):
    
    """Cleans aggregated data and saves it to <processed_path>."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    interim_path = os.path.normpath(interim_path)
    logger.debug("Path to iterim data normalized: {}"
                 .format(interim_path))
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    
    # load aggregated_df
    aggregated_df = pd.read_pickle(os.path.join(interim_path, 'aggregated_df.pkl'))
    logger.info("Loaded aggregated_df.pkl with {} files."
                .format(len(aggregated_df)))
    
    # start cleaning
    start = time()
    
    # split aggregated_df
    teams_df, features_df = aggregated_df.iloc[:,:9], aggregated_df.iloc[:,9:]
    
    # create multi-index
    multi_index = pd.MultiIndex.from_tuples(features_df.columns,
                                            names=['feature', 'stat'])
    features_df.columns = multi_index
    
    # drop mean columns since mostly irrelevant
    df = features_df.drop('mean', axis=1, level=1)
    
    # re-include relevant columns from means
    df = pd.concat([features_df[('radon_avg_cc', 'mean')],
                    features_df[('radon_mi', 'mean')],
                    features_df[('radon_raw_is_error', 'mean')],
                    features_df[('radon_cc_is_error', 'mean')],
                    features_df[('radon_h_is_error', 'mean')],
                    features_df[('radon_mi_is_error', 'mean')],
                    features_df[('pylint_is_error', 'mean')],
                    df], axis=1)
    
    # drop halstead metrics since they are not additive thus wrong
    df = df.drop([('radon_h_calculated_length', 'sum'),
                  ('radon_h_volume', 'sum'),
                  ('radon_h_difficulty', 'sum'),
                  ('radon_h_effort', 'sum'),
                  ('radon_h_time', 'sum'),
                  ('radon_h_bugs', 'sum')], axis=1)
    
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
    
    df = pd.concat([teams_df, df], axis=1)
    
    # correlation coefficient matrix
    corr = df.iloc[:,:40].corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(35, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    svm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    figure = svm.get_figure()    
    figure.savefig('svm_conf.png', dpi=100)
    
#    # seperate sum features and mean features
#    sums = features_df.xs('sum', axis=1, level=1)
#    means = features_df.xs('mean', axis=1, level=1)
    
    # clean data
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to extract the features: {}".format(time_passed))

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
            help = """path to extracted features features_df.pkl
                    (default: data/interim)""")
    parser.add_argument(
            '-p', '--processed_path',
            default = "data/processed",
            help = """path to store the aggregated output repos_df.pkl
                    (default: data/processed)""")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path, args.processed_path)