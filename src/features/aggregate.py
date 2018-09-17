# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import numpy as np


def main(metadata_path = "data/raw/meta-kaggle-2016",
         interim_path = "data/interim",
         processed_path = "data/processed"):
    
    """Aggregates extracted features with Teams.csv to repos_df."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    metadata_path = os.path.normpath(metadata_path)
    logger.debug("Path to meta-kaggle-2016 normalized: {}".format(metadata_path))
    interim_path = os.path.normpath(interim_path)
    logger.debug("Path to iterim data normalized: {}".format(interim_path))
    processed_path = os.path.normpath(processed_path)
    logger.debug("Path to processed data normalized: {}".format(processed_path))
    
    # load Teams.csv
    teams_df = pd.read_csv(os.path.join(metadata_path, 'Teams.csv'), low_memory=False)
    logger.info("Loaded Team.csv with {} repositories.".format(len(teams_df)))
    
    # load features_df
    features_df = pd.read_pickle(os.path.join(interim_path, 'features_df.pkl'))
    logger.info("Loaded Team.csv with {} files.".format(len(features_df)))
    
    # create repos_df by filtering for repos with features
    ids_set = set(features_df['repo_id'])
    in_features = [repo_id in ids_set for repo_id in teams_df['Id'].tolist()]
    repos_df = teams_df[in_features].set_index('Id')
    logger.info("Created repos_df.")


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/extract.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Aggregates extracted features with Teams.csv to repos_df.")
    parser.add_argument(
            '-m', '--metadata_path',
            default = "data/raw/meta-kaggle-2016",
            help = "path to Kaggle Meta Dataset 2016, where Teams.csv is (default: data/raw/meta-kaggle-2016)")
    parser.add_argument(
            '-i', '--interim_path',
            default = "data/interim",
            help = "path to extracted features features_df.pkl (default: data/interim)")
    parser.add_argument(
            '-p', '--processed_path',
            default = "data/processed",
            help = "path to store the aggregated output repos_df.pkl (default: data/processed)")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path)