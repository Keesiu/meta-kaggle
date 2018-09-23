# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main(processed_path = "data/processed",
         models_path = "models"):
    
    """Trains the model."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    models_path = os.path.normpath(models_path)
    logger.debug("Path to models normalized: {}"
                 .format(models_path))
    
    # load df
    df = pd.read_pickle(os.path.join(processed_path, 'df.pkl'))
    logger.info("Loaded df.pkl. Shape of df: {}"
                .format(df.shape))
    
    #%% start training
    start = time()
    
    y, X = np.split(df, [1], axis=1)
    
    X_train, X_test, y_train, y_test  = train_test_split(
            X, y, test_size=0.3, random_state=42)
    
    # logistic regression
    
    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.score(X_test, y_test)
    lr.get_params()
    lr.coef_
    
    plt.plot(X, lr.predict(X))
    plt.show()
    
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
    main(args.processed_path, args.models_path)