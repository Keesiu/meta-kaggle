import os, logging, argparse, re
import pandas as pd
from time import time


def main(repos_path = "data/external/repositories",
         scripts_df_path = "data/interim"):
    
    """Tables all external python scripts in a Pandas DataFrame scripts_df."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize path
    repos_path = os.path.normpath(repos_path)
    logger.debug("Path to repositories normalized: {}".format(repos_path))
    scripts_df_path = os.path.normpath(scripts_df_path)
    logger.debug("Path to scripts_df normalized: {}".format(scripts_df_path))
    
    # traverse and list all files
    start = time()
    repo_id = []
    path = []
    name = []
    for dirpath, dirnames, filenames in os.walk(repos_path, topdown=True):
        for filename in filenames:
            name.append(filename)
            path.append(dirpath)
            temp = int(re.search('\d{5,6}', dirpath)[0])
            repo_id.append(temp)
            logger.debug("Finished with: repo_id = {}, path = {}, file = {}"
                         .format(temp, dirpath, filename))
    
    # store into pandas Dataframe and pickle
    scripts_df = pd.DataFrame(data={'repo_id' : repo_id,
                                    'path' : path,
                                    'name' : name})
    scripts_df.to_pickle(scripts_df_path)
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed: {}".format(time_passed))


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/table.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Tables all python scripts in a Pandas DataFrame.")
    parser.add_argument(
            '-r', '--repos_path',
            default = "data/external/repositories",
            help = "path to downloaded repositories (default: data/external/repositories)")
    parser.add_argument(
            '-s', '--scripts_path',
            default = "data/interim",
            help = "path to downloaded repositories (default: data/interim)")
    args = parser.parse_args()
    
    # run main
    main(args.repos_path, args.scripts_path)