import os, logging, argparse, re
import pandas as pd
from time import time


def main(repos_path = "data/external/repositories",
         scripts_df_path = "data/interim"):
    
    """Tables all external python scripts in a Pandas DataFrame scripts_df.
    
    DataFrame scripts_df has 4 columns:
    repo_id: the respective repository ID of the script from Team.csv
    path: path to the script file
    name: name to the script file
    content: context of that script file as string"""
    
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
            logger.debug("Tracked: repo_id = {:6}, path = ...{:54} file = {}."
                         .format(temp, dirpath[-50:], filename))
    logger.info("Tracked {} files.".format(len(name)))
    
    # read content
    content = []
    for i in range(len(name)):
        try:
            f_path = os.path.join(path[i], name[i])
            with open(f_path) as f:
                temp = f.read()
            logger.debug("Successfully opened file: {}.".format(f_path))
        except Exception:
            logger.exception("Failed to open file:  {}.".format(f_path))
        content.append(temp)
    
    # store into pandas Dataframe and pickle
    scripts_df = pd.DataFrame(data={'repo_id' : repo_id,
                                    'path' : path,
                                    'name' : name,
                                    'content' : content})
    logger.info("Created script_df.")
    scripts_df.to_pickle(os.path.join(scripts_df_path, 'scripts_df.pkl'))
    logger.info("Saved script_df to {}."
                .format(os.path.join(scripts_df_path, 'scripts_df.pkl')))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to table the scripts: {}".format(time_passed))


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