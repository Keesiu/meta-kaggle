import requests, zipfile, io, os, logging, argparse
import pandas as pd

def main(team_csv_path, output_path):
    
    # logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
                        filename='logs/download.log',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    
    # load Team.csv
    team_csv = pd.read_csv(team_csv_path, low_memory=False)
    team_csv.index = team_csv.index.map(str)
    logger.info('Loaded repo list with {} entries'.format(len(team_csv)))
    # drop all entries, that don't have a repo entry
    team_csv = team_csv.dropna(subset=['GithubRepoLink'])
    logger.info('There are {} entries with a repo link'.format(len(team_csv)))
    
    i = 0
    for _, row in team_csv.iterrows():
        
        # skip if the current folder already exists
        repo_path = os.path.join(output_path, str(row['Id']))
        if os.path.exists(repo_path):
            logger.info('Repo {} already exists'.format(str(row['Id'])))
            continue
        
        # try downloading repositories
        repo_url = row['GithubRepoLink']
        repo_zip_url = repo_url + '/archive/master.zip'
        try:
            logger.info('Downloading repo of Team {} with id {!s}'.format(row['TeamName'], str(row['Id'])))
            req = requests.get(repo_zip_url, allow_redirects=True)
            req.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(req.content))
            z.extractall(repo_path)
            logger.info('Sucessfully extracted {} repo'.format(row['TeamName']))
            i += 1
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.error(e)
        if i == 2: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads available Github repositories from Team.csv.')
    parser.add_argument('team_csv_path',
                        help='path to Teams.csv')
    parser.add_argument('output_path',
                        help='path where downloaded repositories should be stored')
    args = parser.parse_args()
    main(args.team_csv_path, args.output_path)