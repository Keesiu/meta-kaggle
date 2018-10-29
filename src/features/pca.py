# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main(processed_path = "data/processed"):
    
    """Performs PCA on the file '<df_name>_df.pkl'.
    
    Input:
        <processed_path> = Path to read input and to store output
        <df_name> = name of DataFrame
    Output:
        Saves PCA-transformed data '<df_name>_pca_df.pkl' to <processed_path>
        and a DataFrame '<df_name>_pca_components_df.pkl' to <processed_path>
    """
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    
    # load selected_df
    selected_df = pd.read_pickle(os.path.join(processed_path,
                                              'selected_df.pkl'))
    logger.info("Loaded selected_df. Shape: {}"
                .format(selected_df.shape))
    
    #%% split df into dependent and independent variables
    teams_df = selected_df.iloc[:, :9]
    y = selected_df.iloc[:, 9:11]
    X = selected_df.iloc[:, 11:]
    X_columns = X.columns
    X_index = X.index
    start = time()

    #%% standardize X and perform PCA
    
#    # normalize
#    for col in ['radon_cc_mean', 'radon_mi_mean', 'loc_max']:
#        print(col, min(X[col]), max(X[col]))
#        X[col] -= min(X[col])
#        X[col] /= max(X[col])
    
    logger.info("Before PCA:\n{}".format(X.describe()))
    
    # standardize
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X.values),
                     index = X_index,
                     columns = X_columns)
    logger.info("After Standardization:\n{}".format(X.describe()))
    
    # plot curve to find suitable number of components
    pca = PCA().fit(X.values)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.axis([0, 80, 0, 1])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    
    # manually set suitable number of components
    N_COMPONENTS = 10
    logger.debug("Constant N_COMPONENTS is set to {}.".format(N_COMPONENTS))
    
    # PCA
    pca = PCA(n_components=N_COMPONENTS)
    pca = pca.fit(X.values)
    pca_components_df = pd.DataFrame(pca.components_, columns=X_columns)
    X = pd.DataFrame(pca.transform(X.values),
                     index = X_index)
    logger.info("After PCA:\n{}".format(X.describe()))
    logger.info("{} principal components explain {:05.2f}% of total variance."
                .format(N_COMPONENTS, sum(pca.explained_variance_ratio_)*100))
    
    pca_df = pd.concat([y, X], axis=1)
    
    #%% export pca_df as pickle file to processed folder
    pca_df.to_pickle(os.path.join(processed_path, 'pca_df.pkl'))
    logger.info("Saved pca_df to {}."
            .format(os.path.join(processed_path, 'pca_df.pkl')))
    
    #%% export pca_components_df as pickle file to processed folder
    logger.info("Components:\n{}".format(pca_components_df))
    pca_components_df.to_pickle(os.path.join(processed_path,
                                             'pca_components_df.pkl'))
    logger.info("Saved pca_components_df to {}."
            .format(os.path.join(processed_path, 'pca_components_df.pkl')))
    
    #%% logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to perform PCA: {}".format(time_passed))
    
#%%
if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/pca.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Performs PCA.")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to load the cleaned data df.pkl \
                    (default: data/processed)")
    parser.add_argument(
            '--df_name',
            default = "selected",
            help = "df to perform PCA on (either 'cleaned' or 'selected') \
                    (default: selected)")
    args = parser.parse_args()
    
    # run main
    main(args.processed_path, args.df_name)