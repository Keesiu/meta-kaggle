# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt


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
    [team_df, features_df] = np.split(aggregated_df, [9], axis=1)
    # create multi-index
    multi_index = pd.MultiIndex.from_tuples(features_df.columns, names = ['feature', 'stat'])
    features_df.columns = multi_index
    sums = features_df.xs('sum', axis=1, level=1)
    means = features_df.xs('mean', axis=1, level=1)
    
    # correlation coefficient matrix
    temp = aggregated_df.corr()
    plt.matshow(aggregated_df.corr())
    
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