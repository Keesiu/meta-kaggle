# -*- coding: utf-8 -*-

import os, sys, logging, argparse

# forces import statement to also search in cwd (should be .../meta-kaggle)
# see: chrisyeh96.github.io/2017/08/08/
# definitive-guide-python-imports.html#more-on-syspath
if '' not in sys.path:
    sys.path.insert(0, '')

from src.data import download, reduce, translate2to3, table
from src.features import extract, aggregate, clean, select, pca
from src.models import train

def main(metadata_path, repos_path, interim_path, processed_path, models_path):
    
    """Runs everything, and skips steps already done."""
    
    # normalize paths
    metadata_path = os.path.normpath(metadata_path)
    logging.debug("Path to metadata normalized: {}".format(metadata_path))
    repos_path = os.path.normpath(repos_path)
    logging.debug("Path to repositories normalized: {}".format(repos_path))
    interim_path = os.path.normpath(interim_path)
    logging.debug("Path to interim data normalized: {}".format(interim_path))
    processed_path = os.path.normpath(processed_path)
    logging.debug("Path to processed data normalized: {}"
                  .format(processed_path))
    
    # downloads Github repos from Team.csv to data/external/repositories
    if not os.path.exists(repos_path + '_2to3'):
        logging.info("Starting download.py.")
        download.main(metadata_path, repos_path)
        logging.info("Finished download.py.")
    
    # reduces Github repositories data by deleting every non-Python file
    if not os.path.exists(repos_path + '_2to3'):
        logging.info("Starting reduce.py.")
        reduce.main(repos_path)
        logging.info("Finished reduce.py.")
    
    # translates the external Python scripts from version 2.x to 3.x
    if not os.listdir(repos_path) == os.listdir(repos_path + '_2to3'):
        logging.info("Starting translate2to3.py.")
        translate2to3.main(repos_path)
        logging.info("Finished translate2to3.py.")
    
    # tables the 2to3-translated scripts to tabled_df
    if not os.path.isfile(os.path.join(interim_path, 'tabled_df.pkl')):
        logging.info("Starting table.py.")
        table.main(repos_path, interim_path)
        logging.info("Finished table.py.")
    
    # extracts source code metrics from script content to extracted_df
    if not os.path.isfile(os.path.join(interim_path, 'extracted_df.pkl')):
        logging.info("Starting extract.py.")
        extract.main(interim_path)
        logging.info("Finished extract.py.")
    
    # aggregates extracted features to repo level and saves to aggregated_df
    if not os.path.isfile(os.path.join(interim_path, 'aggregated_df.pkl')):
        logging.info("Starting aggregate.py.")
        aggregate.main(metadata_path, interim_path)
        logging.info("Finished aggregate.py.")
    
    # cleans aggregated features to become meaningful and saves to cleaned_df
    if not os.path.isfile(os.path.join(processed_path, 'cleaned_df.pkl')):
        logging.info("Starting clean.py.")
        clean.main(interim_path, processed_path)
        logging.info("Finished clean.py.")
    
    # selects relevant cleaned features for modeling and saves to selected_df
    if not os.path.isfile(os.path.join(processed_path, 'selected_df.pkl')):
        logging.info("Starting select.py.")
        select.main(processed_path)
        logging.info("Finished select.py.")
    
    # tranforms selected_df with PCA and saves to pca_df
    # additionally, saves the respective PCA components
        if not os.path.isfile(os.path.join(processed_path, 'pca_df.pkl')):
            logging.info("Starting pca.py for selected_df.")
            pca.main(processed_path)
            logging.info("Finished pca.py, created pca_df.pkl.")
    
    # trains the elastic net regression on all possible combinations
    for y_name in ['ranking_log', 'score']:
        if not os.path.isfile(os.path.join(
                models_path, y_name + '_sm_OLS_fit_regularized_summary.txt')):
            logging.info("Starting train.py for y_name='{}'.".format(y_name))
            train.main(processed_path, models_path, y_name)
            logging.info("Finished train.py for y_name='{}'.".format(y_name))

if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/runall.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Runs everything.")
    parser.add_argument(
            '--metadata_path',
            default = "data/raw/meta-kaggle-2016",
            help = "path to Kaggle Meta Dataset 2016 \
                    (default: data/raw/meta-kaggle-2016)")
    parser.add_argument(
            '--repos_path',
            default = "data/external/repositories",
            help = "path to downloaded repositories \
                    (default: data/external/repositories)")
    parser.add_argument(
            '--interim_path',
            default = "data/interim",
            help = "path to interim data (default: data/interim)")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to processed data (default: data/processed)")
    parser.add_argument(
            '--models_path',
            default = "models",
            help = "path to the trained models (default: models)")
    args = parser.parse_args()
    
    # run main
    main(args.metadata_path,
         args.repos_path,
         args.interim_path,
         args.processed_path,
         args.models_path)