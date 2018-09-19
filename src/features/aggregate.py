# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import collections
from time import time


def main(metadata_path = "data/raw/meta-kaggle-2016",
         interim_path = "data/interim"):
    
    """Aggregates extracted features to repos_df.
    
    Builds teams_df from Teams.csv by dropping all repos without python files,
    aggregates all source code metrics by repository ID and compute basic
    statistical metrics per repository, like mean and sum. Finally, combines
    these statistics with teams_df and save it to <interim_path>."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    metadata_path = os.path.normpath(metadata_path)
    logger.debug("Path to meta-kaggle-2016 normalized: {}"
                 .format(metadata_path))
    interim_path = os.path.normpath(interim_path)
    logger.debug("Path to iterim data normalized: {}"
                 .format(interim_path))
    
    # load Teams.csv
    teams_df = pd.read_csv(os.path.join(metadata_path, 'Teams.csv'),
                           low_memory=False)
    logger.info("Loaded Team.csv with {} repositories."
                .format(len(teams_df)))
    
    # load features_df
    features_df = pd.read_pickle(os.path.join(interim_path, 'features_df.pkl'))
    logger.info("Loaded features_df.pkl with {} files."
                .format(len(features_df)))
    
    # start aggregation
    start = time()
    
    # reduces teams_df by filtering for repos with features
    ids_counter = collections.Counter(features_df['repo_id'])
    ids_set = set(ids_counter)
    in_features = [repo_id in ids_set for repo_id in teams_df['Id'].tolist()]
    teams_df = teams_df[in_features].set_index('Id')
    logger.info("Created repos_df.")
    
    # add column 'n_scripts' which counts the number of scripts
    teams_df['n_scripts'] = [ids_counter[key] for key in teams_df.index]
    logger.debug("Added column 'n_scripts' which is the number of scripts.")
    
    # transform boolean columns (like _is_error flags) to integer
    n_bool = len(features_df.select_dtypes('bool').columns)
    for col in features_df.select_dtypes('bool').columns:
        features_df[col] = features_df[col].astype(int)
    logger.debug("Trasformed {} boolean columns to integer."
                 .format(n_bool))
    
    # splits features_df along 'repo_id'
    grouped = features_df.groupby('repo_id')
    logger.debug("Splitted features_df into groups along repo_id.")
    
    # aggregates features
    aggregated_df = grouped.describe()
    logger.info("{} features aggregated to {} statistical features (max = {})"
            .format(features_df.shape[1]-1,
                    aggregated_df.shape[1],
                    8*(features_df.shape[1]-1)))
    
    # concatenates teams_df with aggregated_df
    aggregated_df = pd.concat([teams_df, aggregated_df], axis=1)
    logger.info("Combined aggregated features with repos_df.")
    
    # export aggregated_df as pickle file to processed folder
    aggregated_df.to_pickle(os.path.join(interim_path, 'aggregated_df.pkl'))
    logger.info("Saved repos_df to {}."
            .format(os.path.join(interim_path, 'aggregated_df.pkl')))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to extract the features: {}".format(time_passed))


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/aggregate.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Aggregates extracted features to repos_df.")
    parser.add_argument(
            '-m', '--metadata_path',
            default = "data/raw/meta-kaggle-2016",
            help = """path to Kaggle Meta Dataset 2016, where Teams.csv is
                    (default: data/raw/meta-kaggle-2016)""")
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
    main(args.metadata_path, args.interim_path, args.processed_path)