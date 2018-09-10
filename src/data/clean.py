import os, logging, argparse
import pandas as pd
from time import time


def main(repos_path = "data/external/repositories"):
    
    """Cleans the downloaded repositories.
    
    First, traverses all files and delete those not ending on '.py'.
    Then, deletes all empty folders, also empty repository folders.
    """
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize path
    repos_path = os.path.normpath(repos_path)
    logger.debug("Path to repositories normalized: {}".format(repos_path))
    
    start = time()
    n_repos = len(os.listdir(repos_path))
    n_deleted_files = 0
    n_failed_files = 0
    n_deleted_folders = 0
    n_failed_folders = 0
    n_deleted_repos = 0
    
    # traverse bottom-up
    for dirpath, dirnames, filenames in os.walk(repos_path, topdown=False):
        # delete file if not .py file
        for filename in filenames:
            if filename[-3:] != '.py':
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    logger.debug("Successfully deleted file: {}"
                                 .format(file_path))
                    n_deleted_files += 1
                except Exception:
                    logger.exception("Failed to delete file:     {}"
                                     .format(file_path))
                    n_failed_files += 1
        # delete empty folders
        for dirname in dirnames:
            path = os.path.join(dirpath, dirname)
            if not os.listdir(path):
                try:
                    os.rmdir(path)
                    logger.debug(logger.info("Successfully deleted folder: {}"
                                             .format(path)))
                    n_deleted_folders += 1
                    if dirpath == repos_path:
                        n_deleted_repos += 1
                except Exception:
                    logger.exception("Failed to delete folder:     {}"
                                     .format(path))
                    n_failed_folders += 1

    logger.info("Successfully deleted {} files, {} errors occurred."
                .format(n_deleted_files, n_failed_files))
    logger.info("Afterwards, {} empty folders were deleted, {} errors occured."
                .format(n_deleted_folders, n_failed_folders))
    logger.info("From {} repositories, {} were deleted, {} remaining."
                .format(n_repos, n_deleted_repos, len(os.listdir(repos_path))))
        
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed: {}".format(time_passed))


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/clean.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Deletes all non-python files and empty folders.")
    parser.add_argument(
            '-r', '--repos_path',
            default = "data/external/repositories",
            help = "path to downloaded repositories (default: data/external/repositories)")
    args = parser.parse_args()
    
    # run main
    main(args.repos_path)