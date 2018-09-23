# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time


def main(processed_path = "data/processed",
         models_path = "models"):
    
    """Trains the model."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    
    # load aggregated_df
    df = pd.read_pickle(os.path.join(processed_path, 'df.pkl'))
    logger.info("Loaded df.pkl. Shape of df: {}"
                .format(df.shape))
    
    #%% start training
    start = time()
    
    
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
            help = "path to load the cleaned data df.pkl \
                    (default: data/processed)")
        parser.add_argument(
            '--models_path',
            default = "models",
            help = "path to save the trained models \
                    (default: models)")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path, args.processed_path)