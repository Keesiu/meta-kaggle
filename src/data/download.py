import requests, zipfile, io, os, logging, argparse
import pandas as pd
from time import time

def main(team_csv_path="data/raw/meta-kaggle-2016/Teams.csv", output_path="data/external/repositories"):
    
    # logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
                        filename="logs/download.log",
                        datefmt="%a, %d %b %Y %H:%M:%S")
    
    # load Teams.csv
    team_csv = pd.read_csv(team_csv_path, low_memory=False)
    n_total = len(team_csv)
    logger.info("Loaded Team.csv with {} entries.".format(n_total))
    
    # drop all entries, that don't have a repo entry
    team_csv = team_csv.dropna(subset=['GithubRepoLink'])
    n_link = len(team_csv)
    logger.info("In Team.csv from {} entries {} have a GithubRepoLink.".format(n_total, n_link))
    
    i = 0
    n_try = 0
    n_success = 0
    n_error = 0
    start = time()
    for _, row in team_csv.iterrows():
        
        # skip if the current folder already exists
        repo_path = os.path.join(output_path, str(row['Id']))
        if os.path.exists(repo_path):
            logger.info("---- Download of repo {:6} {:11} Team: {:20.15}"
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
            logger.info("---- Download of repo {:6} {:11} Team: {:20.15}"
                        .format(row['Id'], 'successful.', row['TeamName']+'.'))
            i += 1
            n_success += 1
        except (requests.exceptions.RequestException,
                requests.exceptions.HTTPError) as e:
            logger.error("---- Download of repo {:6} {:11} Team: {:20.15} {}"
                        .format(row['Id'], 'failed.', row['TeamName']+'.', e))
            n_error += 1
        if i == 5: break
    end = time()
    logger.info("Successfully downloaded {} repos from {} tries ({} errors). Time needed: {}"
                .format(n_success, n_try, n_error, pd.Timedelta(seconds=end-start).round(freq='s')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Downloads available Github repositories from Team.csv.")
    parser.add_argument('-i', '--input_path',
                        default="data/raw/meta-kaggle-2016/Teams.csv",
                        help="path to Teams.csv (default: data/raw/meta-kaggle-2016/Teams.csv)")
    parser.add_argument('-o', '--output_path',
                        default="data/external/repositories",
                        help="path to store outputs (default: data/external/repositories)")
    args = parser.parse_args()
    main(args.input_path, args.output_path)