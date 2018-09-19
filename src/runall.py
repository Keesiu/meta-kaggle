# -*- coding: utf-8 -*-

import sys, logging, argparse

# forces import statement to also search in cwd (should be .../meta-kaggle)
# see: chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html#more-on-syspath
if '' not in sys.path:
    sys.path.insert(0, '')

from src.data import download, reduce, table
from src.features import extract, aggregate


def main(metadata_path, repos_path, interim_path, processed_path):
    
    """Runs everything.
    
    Downloads external data from Teams.csv and cleans it.
    """
    
    # downloads Github repositories from Team.csv to data/external/repositories
    logging.info("Starting download.py.")
    download.main(metadata_path, repos_path)
    logging.info("Finished download.py.")
    
    # cleans Github repositories by deleting every non-Python file
    logging.info("Starting reduce.py.")
    reduce.main(repos_path)
    logging.info("Finished reduce.py.")
    
    # tables the external scripts and their content to scripts_df
    logging.info("Starting table.py.")
    table.main(repos_path, interim_path)
    logging.info("Finished table.py.")
    
    # extracts source code metrics from script content to features_df
    logging.info("Starting extract.py.")
    extract.main(interim_path)
    logging.info("Finished extract.py.")
    
    # aggregate source code metrics from features_df to repos_df
    logging.info("Starting aggregate.py.")
    aggregate.main(metadata_path, interim_path, processed_path)
    logging.info("Finished aggregate.py.")


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
            '-m', '--metadata_path',
            default = "data/raw/meta-kaggle-2016",
            help = "path to Kaggle Meta Dataset 2016, where Teams.csv is (default: data/raw/meta-kaggle-2016)")
    parser.add_argument(
            '-r', '--repos_path',
            default = "data/external/repositories",
            help = "path to store downloaded repositories (default: data/external/repositories)")
    parser.add_argument(
            '-i', '--interim_path',
            default = "data/interim",
            help = "path to store the output features_df.pkl (default: data/interim)")
    parser.add_argument(
            '-p', '--processed_path',
            default = "data/processed",
            help = "path to store the aggregated output repos_df.pkl (default: data/processed)")
    args = parser.parse_args()
    
    # run main
    main(args.metadata_path, args.repos_path, args.interim_path, args.processed_path)