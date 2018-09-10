import sys, logging, argparse

# forces import statement to also search in cwd (should be .../meta-kaggle)
# see: chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html#more-on-syspath
if '' not in sys.path:
    sys.path.insert(0, '')

from src.data import download
from src.data import clean
from src.features import table


def main(teams_path, repos_path):
    
    """Runs everything.
    
    Downloads external data from Teams.csv and cleans it.
    """
    
    # download Github repositories from Team.csv to data/external/repositories
    logging.info("Starting download.py.")
    download.main(teams_path, repos_path)
    logging.info("Finished download.py.")
    
    # clean Github repositories by deleting every non-Python file
    logging.info("Starting clean.py.")
    clean.main(repos_path)
    logging.info("Finished clean.py.")
    
    # tables the external scripts in a DataFrame
    logging.info("Starting table.py.")
    table.main(repos_path)
    logging.info("Finished table.py.")


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/runall.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Runs everything.")
    parser.add_argument(
            '-t', '--teams_path',
            default = "data/raw/meta-kaggle-2016/Teams.csv",
            help = "path to Teams.csv (default: data/raw/meta-kaggle-2016/Teams.csv)")
    parser.add_argument(
            '-r', '--repos_path',
            default = "data/external/repositories",
            help = "path to store downloaded repositories (default: data/external/repositories)")
    args = parser.parse_args()
    
    # run main
    main(args.teams_path, args.repos_path)