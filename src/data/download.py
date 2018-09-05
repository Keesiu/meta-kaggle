import requests, zipfile, io, os, logging, argparse
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

def main(args):
    
    github_list = pd.read_csv(args.repo_list_path, low_memory=False)
    github_list.index = github_list.index.map(str)
    logger.info('Loaded repo list with {} entries'.format(len(github_list)))
    
    # drop all entries, that don't have a repo entry
    github_list = github_list.dropna(subset=['GithubRepoLink'])
    logger.info('There are {} entries with a repo link'.format(len(github_list)))
    i = 0
    for _, row in github_list.iterrows():
        # check if the folder already exists
        final_download_path = os.path.join(args.output_path, str(row['Id']))
        if os.path.exists(final_download_path):
            logger.info('Repo {} already exists'.format(str(row['Id'])))
            continue
    
        repo_url = row['GithubRepoLink']
        repo_zip_url = repo_url + '/archive/master.zip'
        try:
            logger.info('Downloading repo of Team {} with id {!s}'.format(row['TeamName'], str(row['Id'])))
            req = requests.get(repo_zip_url, allow_redirects=True)
            req.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(req.content))
            z.extractall(final_download_path)
            logger.info('Sucessfully extracted {} repo'.format(row['TeamName']))
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.error(e)
        i += 1
        if i == 5: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack Github repos from a csv file')
    parser.add_argument('repo_list_path', help='path to the list of github repos')
    parser.add_argument('output_path', help='path where the downloaded repos will be stored')
    args = parser.parse_args()
    main(args)