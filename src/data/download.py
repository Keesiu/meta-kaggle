# -*- coding: utf-8 -*-

import requests, zipfile, io, os, logging, argparse
import pandas as pd
from time import time


def main(metadata_path = "data/raw/meta-kaggle-2016",
         repos_path = "data/external/repositories"):
    
    """Downloads the available repositories from Teams.csv.
    
    Iterates GithubRepoLink column of Teams.csv and downloads the master branch
    if available. Folders already existing on <repos_path> are skipped.
    """
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize paths
    metadata_path = os.path.normpath(metadata_path)
    logger.debug("Path to Team.csv normalized: {}".format(metadata_path))
    repos_path = os.path.normpath(repos_path)
    logger.debug("Path to repositories normalized: {}".format(repos_path))
    
    # load Teams.csv
    team_csv = pd.read_csv(os.path.join(metadata_path, 'Teams.csv'), low_memory=False)
    n_total = len(team_csv)
    logger.info("Loaded Team.csv with {} entries.".format(n_total))
    
    # drop all entries, that don't have a repo entry
    team_csv_with_link = team_csv.dropna(subset=['GithubRepoLink'])
    n_link = len(team_csv_with_link)
    logger.info("In Team.csv from {} entries {} have a GithubRepoLink."
            .format(n_total, n_link))
    
    n_try = 0
    n_success = 0
    n_error = 0
    start = time()
    for _, row in team_csv_with_link.iterrows():
        
        # skip if the current folder already exists
        repo_path = os.path.join(repos_path, str(row['Id']))
        if os.path.exists(repo_path):
            logger.warning("Download of repo {:6} {:11} Team: {:20.15}"
                    .format(row['Id'], 'skipped.', row['TeamName']+'.'))
            continue
        
        # try to download current repository
        repo_url = row['GithubRepoLink']
        repo_zip_url = repo_url + "/archive/master.zip"
        try:
            n_try += 1
            req = requests.get(repo_zip_url, allow_redirects=True)
            req.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(req.content))
            z.extractall(repo_path)
            logger.debug("Downloading repo {:6} successful. Team: {:20.15}."
                    .format(row['Id'], row['TeamName']))
            n_success += 1
        except Exception:
            logger.exception("Downloading repo {:6} failed.     Team: {:20.15}."
                    .format(row['Id'], row['TeamName']))
            n_error += 1
    
    logger.info("Successfully downloaded {} repos from {} tries ({} errors)."
                .format(n_success, n_try, n_error))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed: {}".format(time_passed))


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/download.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Downloads available Github repositories from Team.csv.")
    parser.add_argument(
            '-t', '--metadata_path',
            default = "data/raw/meta-kaggle-2016",
            help = "path to Kaggle Meta Dataset 2016, where Teams.csv is (default: data/raw/meta-kaggle-2016)")
    parser.add_argument(
            '-r', '--repos_path',
            default = "data/external/repositories",
            help = "path to store downloaded repositories (default: data/external/repositories)")
    args = parser.parse_args()
    
    # run main
    main(args.metadata_path, args.repos_path)