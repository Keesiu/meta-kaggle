# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


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
    
    # load selected_df
    selected_df = pd.read_pickle(os.path.join(processed_path, 'selected_df.pkl'))
    logger.info("Loaded selected_df.pkl. Shape of selected_df: {}"
                .format(selected_df.shape))
    
    # split df into dependent and independent variables
    y, X = np.split(selected_df, [2], axis=1)
    
    #%% PCA
    
    # normalize
    for col in ['radon_cc_mean', 'radon_mi_mean', 'loc_max']:
        print(col, min(X[col]), max(X[col]))
        X[col] -= min(X[col])
        X[col] /= max(X[col])
    
    # standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    temp = pd.DataFrame(X).describe()
    
    pca = PCA(n_components=10)
    X = pca.fit_transform(X)
    pca.components_
    pca.explained_variance_
    pca.explained_variance_ratio_
    
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.axis([0, 80, 0, 1])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    
    #%% start training
    start = time()
    
#    # train-test-split
#    X_train, X_test, y_train, y_test  = train_test_split(
#            X, y, test_size=0.3, random_state=42)
    
#    # Linear regression with sklearn
#    lr = LinearRegression()
#    lr.fit(X_train, y_train)
#    lr.score(X_test, y_test)
#    lr.get_params()
#    lr.coef_
#    plt.plot(X, lr.predict(X))
#    plt.show()
    
    # Linear regression with statsmodels
    mod = sm.OLS(y.ranking_log, sm.add_constant(X))
    res = mod.fit()
    print(res.summary())
    
    # Elastic net linear regression with statsmodels
    mod = sm.OLS(y.ranking_log, sm.add_constant(X))
    res = mod.fit_regularized(method='elastic_net', alpha=0.1, L1_wt=1.0, refit=True)
    params = res.params
    print(res.summary())
    
    # logistic regression with statsmodels
    mod = sm.Logit(y.score, sm.add_constant(X))
    res = mod.fit()
    print(res.summary())
    
    # lasso logistic regression with statsmodels
    mod = sm.Logit(y.score, sm.add_constant(X))
    res = mod.fit_regularized(method='l1', alpha=.62)
    print(res.summary())
    
    # LR with statsmodels in R-style
    model = ols("Ranking ~ radon_sum_cc_ratio + pylint_class_ratio", selected_df)
    results = model.fit()
    results.summary()
    
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
            help = "path to load the selected data selected_df.pkl \
                    (default: data/processed)")
    parser.add_argument(
            '--models_path',
            default = "models",
            help = "path to save the trained models \
                    (default: models)")
    args = parser.parse_args()
    
    # run main
    main(args.processed_path, args.models_path)
