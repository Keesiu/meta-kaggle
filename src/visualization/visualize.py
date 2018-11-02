# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import AlphaSelection

def main(processed_path = "data/processed",
         models_path = "models",
         visualizations_path = "visualizations"):
    
    """Creates visualizations."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}"
                 .format(processed_path))
    models_path = os.path.normpath(models_path)
    logger.debug("Path to models normalized: {}"
                 .format(models_path))
    visualizations_path = os.path.normpath(visualizations_path)
    logger.debug("Path to visualizations normalized: {}"
                 .format(visualizations_path))
    
    #%% load selected_df
    selected_df = pd.read_pickle(os.path.join(processed_path,
                                              'selected_df.pkl'))
    logger.info("Loaded selected_df. Shape of df: {}"
                .format(selected_df.shape))
    
    # load models
    mod = pickle.load(open(
            os.path.join(models_path, 'sklearn_ElasticNetCV.pkl'), 'rb'))
    mod_sm = pickle.load(open(
            os.path.join(models_path, 'sm_OLS_fit_regularized.pkl'), 'rb'))

    #%% split selected_df into dependent and independent variables
    teams_df = selected_df.iloc[:, :9]
    y = selected_df.iloc[:, 9:10]
    X = selected_df.iloc[:, 10:]
    yX = pd.concat([y, X], axis=1)
    
    #%% start visualization
    
    start = time()
    sns.set_context('paper')
    rcParams.update({'figure.autolayout': True})
    
    #%% correlation coefficient matrix
    
    corr = yX.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    fig = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                      center=0, square=True, linewidths=.5,
                      cbar_kws={"shrink": .5}).get_figure()
    fig.savefig(os.path.join(visualizations_path,
                             'correlation_coefficient_matrix.png'), dpi=300)
    fig.clear()
    plt.close()
    
    #%% histograms of transformation
    
    sns.set_style("darkgrid")
    
    # histogram of ranking
    fig = sns.distplot(teams_df.Ranking, rug=True,
                       axlabel='ranking').get_figure()
    fig.savefig(os.path.join(visualizations_path,
                             'histogram_ranking.png'), dpi=300)
    fig.clear()
    plt.close()
    
    # histogram of ranking_log
    fig = sns.distplot(y, rug=True, axlabel='ranking_log').get_figure()
    fig.savefig(os.path.join(visualizations_path,
                             'histogam_ranking_log.png'), dpi=300)
    fig.clear()
    plt.close()
    
    # histogram of loc_max
    fig = sns.distplot(np.e**X.loc_max_log, rug=True,
                       axlabel='loc_max').get_figure()
    fig.savefig(os.path.join(visualizations_path,
                             'histogram_loc_max.png'), dpi=300)
    fig.clear()
    plt.close()
    
    # histogram of loc_max_log
    fig = sns.distplot(X.loc_max_log, rug=True,
                       axlabel='loc_max_log').get_figure()
    fig.savefig(os.path.join(visualizations_path,
                         'histogram_loc_max_log.png'), dpi=300)
    fig.clear()
    plt.close()
    
    #%% standardize
    
    scaler = StandardScaler()
    not_standardize = ['core',
                       'visualization',
                       'machine_learning',
                       'deep_learning']
    X_standardized = scaler.fit_transform(X
                                          .drop(columns=not_standardize)
                                          .values)
    X_standardized = pd.DataFrame(X_standardized,
                                  index = X.index,
                                  columns = X.columns.drop(not_standardize))
    X_not_standardized = X[not_standardize]
    X = pd.concat([X_standardized, X_not_standardized], axis=1)
    logger.debug("After Standardization:\n{}".format(X.describe().to_string))
    # update yX
    yX = pd.concat([y, X], axis=1)
    
    #%% boxplot
    
    f, ax = plt.subplots(figsize=(12, 8))
    fig = sns.boxplot(data=yX)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=270)
    fig.get_figure().savefig(os.path.join(visualizations_path,
                                          'boxplot.png'), dpi=300)
    fig.clear()
    plt.close()
    
    #%% residual plot
    f, ax = plt.subplots(figsize=(5, 5))
    fig = sns.residplot(x=mod_sm.fittedvalues, y=y, data=X).get_figure()
    fig.savefig(os.path.join(visualizations_path, 'residplot.png'), dpi=300)
    fig.clear()
    plt.close()

    #%% plot ElasticNetCV results
    
    # need to fix l1_ratio from list to best_l1_ratio
    # in order to visualize correctly
    mod.set_params(l1_ratio=mod.l1_ratio_)
    
    # print MSE's across folds
    m_log_alphas = mod.alphas_
    fig = plt.figure()
    plt.plot(m_log_alphas, mod.mse_path_, ':')
    plt.plot(m_log_alphas, mod.mse_path_.mean(axis=-1), 'b',
                   label='Average over the folds')
    plt.axvline(mod.alpha_, linestyle='--', color='k',
                      label="$\\alpha={:0.3f}$".format(mod.alpha_))
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('error (or score)')
    plt.title('ElasticNetCV Alpha Error (per CV-fold)')
    plt.axis('tight')
    fig.savefig(os.path.join(visualizations_path,
                             'ElasticNetCV_MSE_per_fold.png'), dpi=300)
    fig.clear()
    plt.close()
    
    # print R^2 errors (minimization equivalent to MSE)
    visualizer = AlphaSelection(mod)
    visualizer.fit(X, y)
    visualizer.poof(outpath=os.path.join(visualizations_path,
                                         'ElasticNetCV_MSE.png'), dpi=300)
    plt.close()
    
    #%% pairplot not performed since too big
    
#    X_used = X.loc[:, mod.coef_ != 0]
#    fig = sns.pairplot(pd.concat([y, X_used], axis=1), kind='reg')
#    fig.savefig(os.path.join(visualizations_path,
#                             'pairplot.png'), dpi=100)
#    fig.clear()
#    plt.close()
        
    #%% logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to create visualizations: {}"
                .format(time_passed))
    
#%%

if __name__ == '__main__':

    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/visualize.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Create visualizations.")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to processed data (default: data/processed)")
    parser.add_argument(
            '--models_path',
            default = "models",
            help = "path to the trained models (default: models)")
    parser.add_argument(
            '--visualizations_path',
            default = "visualizations",
            help = "path to the visualizations (default: visualizations)")
    args = parser.parse_args()
    
    # run main
    main(args.processed_path,
         args.models_path,
         args.visualizations_path)