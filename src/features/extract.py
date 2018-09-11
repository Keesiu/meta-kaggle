import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
import radon.raw


def main(interim_path = "data/interim"):
    
    """Extracts all features from scripts_df and saves it in features_df."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # transform path
    interim_path = os.path.normpath(interim_path)
    scripts_df_path = os.path.join(interim_path, 'scripts_df.pkl')
    logger.debug("Path to scripts_df created: {}".format(scripts_df_path))
    
    # load scripts_df and create copy for storing features
    scripts_df = pd.read_pickle(scripts_df_path)
    features_df = scripts_df[['repo_id']].copy()
    logger.info("Created features_df.")

    # start extracting source code metrics
    start = time()
    
    # length of content
    features_df['content_len'] = scripts_df['content'].map(len)
    
    # radon raw metrics
    radon_raw = [try_radon_raw(scripts_df['content'], index, logger) for index in scripts_df.index]
    (features_df['radon_raw_loc'],
     features_df['radon_raw_lloc'],
     features_df['radon_raw_sloc'],
     features_df['radon_raw_comments'],
     features_df['radon_raw_multi'],
     features_df['radon_raw_blank'],
     features_df['radon_raw_single_comments']) = zip(*radon_raw)
    n_radon_raw_error = features_df['radon_raw_loc'].isna().sum()
    n_radon_raw_success = sum(features_df['radon_raw_loc'] >= 0)
    logger.info("Extracted radon raw metrics: {} scripts, {} successes, {} errors."
                .format(len(scripts_df['content']), n_radon_raw_success, n_radon_raw_error))
    
    # store features
    features_df.to_pickle(os.path.join(interim_path, 'features_df.pkl'))
    logger.info("Saved script_df to {}."
                .format(os.path.join(interim_path, 'features_df.pkl')))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to extract the features: {}".format(time_passed))


def try_radon_raw(series, index, logger):
    try:
        result = radon.raw.analyze(series[index])
        logger.debug("Successfully extracted radon raw metrics of file {}."
                     .format(index))
        return list(result)
    except:
        logger.exception("Failed to extract radon raw metrics of file      {}."
                     .format(index))
        return [np.nan]*7


if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/extract.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Extracts all features from scripts_df and saves it in features_df.")
    parser.add_argument(
            '-i', '--interim_path',
            default = "data/interim",
            help = "path to store the output features_df.pkl (default: data/interim)")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path)