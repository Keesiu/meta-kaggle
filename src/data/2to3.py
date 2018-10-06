# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
from time import time
import subprocess


def main(repos_path = "data/external/repositories"):
    
    """Transforms all Python 2.x scripts to 3.x."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize path
    repos_path = os.path.normpath(repos_path)
    logger.debug("Path to repositories normalized: {}".format(repos_path))
    
    
    # start 2to3 transformation
    start = time()
    args = ['2to3',
            '-W',
            '-n',
            '--output-dir='+repos_path+'_2to3',
            repos_path]
    s = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info("Standard output:\n" + s.stdout.decode('latin'))
    logger.info("Standard error:\n" + s.stderr.decode('latin'))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to 2to3-transform the scripts: {}"
                .format(time_passed))


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/2to3.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Transforms all Python 2.x scripts to 3.x.")
    parser.add_argument(
            '--repos_path',
            default = "data/external/repositories",
            help = "path to downloaded repositories \
                    (default: data/external/repositories)")
    args = parser.parse_args()
    
    # run main
    main(args.repos_path)