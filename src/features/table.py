import os, logging, argparse
import pandas as pd
from time import time


def main(repos_path = "data/external/repositories",
         scripts_df_path = "data/interim"):
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize path
    repos_path = os.path.normpath(repos_path)
    logger.debug("Path to repositories normalized: {}".format(repos_path))
    scripts_df_path = os.path.normpath(scripts_df_path)
    logger.debug("Path to scripts_df normalized: {}".format(scripts_df_path))
    
    start = time()
    
    
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed: {}".format(time_passed))


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.INFO,
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