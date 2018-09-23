# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
from time import time


def main(processed_path = "data/processed"):
    
    """Selects features for training the model."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    
    # load df
    df = pd.read_pickle(os.path.join(processed_path, 'df.pkl'))
    logger.info("Loaded df.pkl. Shape of df: {}"
                .format(df.shape))
    
    #%% start feature selection
    start = time()
    
    n = len(df)
    
    selected_df = df.copy()
    
    # drop features with more than 95% zeros
    selected_df.drop(columns = df.columns[(((df == 0).sum()/n) > .95).values],
                     errors = 'ignore',
                     inplace = True)
    
    # drop features with less than 5% unique values
    selected_df.drop(columns = df.columns[(df.nunique()/n < .05).values],
                     errors = 'ignore',
                     inplace = True)
    
        
    # drop features with too little relative standard deviation
    
    # remove multi-collinearity
    
    #%% export selected_df as pickle file to processed folder
    df.to_pickle(os.path.join(processed_path, 'selected_df.pkl'))
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