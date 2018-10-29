# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import collections
from time import time
import itertools


def main(metadata_path = "data/raw/meta-kaggle-2016",
         interim_path = "data/interim"):
    
    """Aggregates extracted features from script to repository level.
    
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
    
    # load extracted_df
    extracted_df = pd.read_pickle(os.path.join(interim_path,
                                               'extracted_df.pkl'))
    logger.info("Loaded extracted_df.pkl with {} files."
                .format(len(extracted_df)))
    
    # start aggregation
    start = time()
    
    # reduces teams_df by filtering for repos with python scripts
    ids_counter = collections.Counter(extracted_df['repo_id'])
    ids_set = set(ids_counter)
    in_features = [repo_id in ids_set for repo_id in teams_df['Id'].tolist()]
    teams_df = teams_df[in_features].set_index('Id')
    logger.info("Reduced teams_df to {} entries with python scripts."
                .format(len(teams_df)))
    
    # add column 'n_scripts' which counts the number of scripts
    teams_df['n_scripts'] = [ids_counter[key] for key in teams_df.index]
    logger.info("Added column 'n_scripts' which is the number of scripts.")
    
    # transform boolean columns (like _is_error flags) to integer
    n_bool = len(extracted_df.select_dtypes('bool').columns)
    for col in extracted_df.select_dtypes('bool').columns:
        extracted_df[col] = extracted_df[col].astype(int)
    logger.info("Turned {} boolean columns of extracted_df to integer."
                 .format(n_bool))
    
    # splits extracted_df along 'repo_id'
    grouped = extracted_df.groupby('repo_id')
    logger.info("Splitted extracted_df into groups along repo_id.")
    
    # aggregate features
    repos = {}
    for repo_id, group in grouped:
        stats = []
        # remove first column 'repo_id'
        for col_name in group.drop('repo_id', axis=1):
            # calculate relevant statistics
            stats.extend([group[col_name].sum(), group[col_name].mean()])
        repos[repo_id] = stats
    columns = list(itertools.product(extracted_df.columns[1:], ['sum','mean']))
    # build aggregated_df from dict repos
    aggregated_df = pd.DataFrame.from_dict(repos,
                                           orient='index',
                                           columns=columns)
    logger.info("{} features on script-level aggregated to {} on repo-level."
                .format(extracted_df.shape[1]-1, aggregated_df.shape[1]))
    
    # concatenates teams_df with aggregated_df
    aggregated_df = pd.concat([teams_df, aggregated_df], axis=1)
    logger.info("Combined aggregated features with teams_df.")
    
    # export aggregated_df as pickle file to interim folder
    aggregated_df.to_pickle(os.path.join(interim_path, 'aggregated_df.pkl'))
    logger.info("Saved aggregated_df to {}."
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
            description = "Aggregates extracted features.")
    parser.add_argument(
            '--metadata_path',
            default = "data/raw/meta-kaggle-2016",
            help = "path to Kaggle Meta Dataset 2016, where Teams.csv is \
                    (default: data/raw/meta-kaggle-2016)")
    parser.add_argument(
            '--interim_path',
            default = "data/interim",
            help = "path to to load extracted_df and store aggregated_df \
                    (default: data/interim)")
    args = parser.parse_args()
    
    # run main
    main(args.metadata_path, args.interim_path)