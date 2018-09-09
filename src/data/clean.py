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
    n_repos = len(os.listdir(repo_path))
    n_deleted_files = 0
    n_failed_files = 0
    n_deleted_folders = 0
    n_failed_folders = 0
    n_deleted_repos = 0
    
    # traverse bottom-up
    for dirpath, dirnames, filenames in os.walk(repo_path, topdown=False):
        # delete file if not .py file
        for filename in filenames:
            if filename[-3:] != '.py':
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    logger.info("{:22}{}"
                                .format("---- Successfully deleted file:", file_path))
                    n_deleted_files += 1
                except Exception:
                    logger.exception("{:22}{}"
                                 .format("---- Failed to delete file: ", file_path))
                    n_failed_files += 1
        # delete empty folders
        for dirname in dirnames:
            path = os.path.join(dirpath, dirname)
            if not os.listdir(path):
                try:
                    os.rmdir(path)
                    logger.info(logger.info("{:22}{}"
                                            .format("---- Successfully deleted folder:", path)))
                    n_deleted_folders += 1
                    if dirpath == repo_path:
                        n_deleted_repos += 1
                except Exception:
                    logger.exception("{:22}{}"
                                 .format("---- Failed to delete folder: ", dirname))
                    n_failed_folders += 1

    logger.info("Successfully deleted {} files, {} errors occurred."
                .format(n_deleted_files, n_failed_files))
    logger.info("Afterwards, {} empty folders were deleted, {} errors occured."
                .format(n_deleted_folders, n_failed_folders))
    logger.info("From {} repositories, {} were deleted, {} remaining."
                .format(n_repos, n_deleted_repos, len(os.listdir(repo_path))))
        
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed: {}"
                .format(time_passed))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description="Deletes all non-python files and empty folders.")
    parser.add_argument(
            '-r', '--repo_path',
            default="data/external/repositories",
            help="path to downloaded repositories (default: data/external/repositories)")
    args = parser.parse_args()
    
    main(args.repo_path)