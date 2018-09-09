import os, logging, argparse
import pandas as pd
from time import time


def main(repo_path = "data/external/repositories"):
    
    # logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            level = logging.INFO,
            format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
            filename = "logs/clean.log",
            datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # normalize path
    repo_path = os.path.normpath(repo_path)
    logger.info("Path to repositories normalized: {}"
                .format(repo_path))
    
    start = time()
    n_deleted = 0
    n_error = 0
    for dirpath, dirnames, filenames in os.walk(repo_path, topdown=True):
        for file in filenames:
            if file[-3:] != '.py':
                try:
                    os.remove(os.path.join(dirpath, file))
                    logger.info("{:22}{}"
                                .format("---- Successfully deleted:", os.path.join(dirpath, file)))
                    n_deleted += 1
                except Exception as e:
                    logger.error("{:22}{}{}"
                                 .format("---- Failed to delete: ", os.path.join(dirpath, file), e))
                    n_error += 1
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Successfully deleted {} files, {} errors occurred. Time needed: {}"
                .format(n_deleted, n_error, time_passed))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description="Cleans downloaded repositories by deleting all non-python files.")
    parser.add_argument(
            '-p', '--repo_path',
            default="data/external/repositories",
            help="path to downloaded repositories (default: data/external/repositories)")
    args = parser.parse_args()
    
    main(args.repo_path)