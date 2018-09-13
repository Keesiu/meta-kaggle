import os, logging, argparse
import pandas as pd
import numpy as np
from time import time
import radon.raw, radon.complexity, radon.metrics


def main(interim_path = "data/interim"):
    
    """Extracts all features from scripts_df and saves it in features_df.
    
    Includes:
        Radon:
        - Raw metrics
        - Cyclomatic Complexity metrics
        - Halstead Metrics
        - Maintainability Index metric
    Every metric type has a flag column <metric type>_is_error indicating
    if an error occurred while trying to extract the respective metrics."""
    
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
    n = len(features_df)
    
    # length of content
    features_df['content_len'] = scripts_df.content.map(len)
    logger.info("Extracted content length 'content_len'.")
    
    # radon raw metrics
    logger.info("Start extracting radon raw metrics.")
    # extraction
    radon_raw = [try_radon_raw(scripts_df.content, index, logger) for index in scripts_df.index]
    # save results to features_df
    (features_df['radon_raw_loc'],
     features_df['radon_raw_lloc'],
     features_df['radon_raw_sloc'],
     features_df['radon_raw_comments'],
     features_df['radon_raw_multi'],
     features_df['radon_raw_blank'],
     features_df['radon_raw_single_comments']) = zip(*radon_raw)
    # set is_error flag
    features_df['radon_raw_is_error'] = features_df.radon_raw_loc.isna()
    # logging results
    n_radon_raw_error = features_df.radon_raw_is_error.sum()
    n_radon_raw_success = sum(features_df.radon_raw_loc >= 0)
    logger.info("Extracted radon raw metrics: {} scripts, {} successes, {} errors."
                .format(n, n_radon_raw_success, n_radon_raw_error))
    
    # radon cyclomatic complexity metrics
    logger.info("Start extracting radon cyclomatic complexity metrics.")
    # extraction
    radon_cc = [try_radon_cc(scripts_df.content, index, logger) for index in scripts_df.index]
    # save results to features_df
    features_df['radon_avg_cc'] = list(map(radon.complexity.average_complexity, radon_cc))
    features_df['radon_sum_cc'] = [sum([obj.complexity for obj in blocks]) for blocks in radon_cc]
    # set is_error flag
    features_df['radon_cc_is_error'] = [isinstance(x, str) for x in radon_cc]
    # logging results
    n_radon_cc_error = sum(features_df.radon_cc_is_error)
    n_radon_cc_success = sum([isinstance(x, list) for x in radon_cc])
    logger.info("Extracted radon cyclomatic complexity metrics: {} scripts, {} successes, {} errors."
                .format(n, n_radon_cc_success, n_radon_cc_error))
    
    # radon halstead metrics
    logger.info("Start extracting radon halstead metrics.")
    # extraction
    radon_h = [try_radon_h(scripts_df.content, index, logger) for index in scripts_df.index]
    # save results to features_df
    (features_df['radon_h_h1'],
     features_df['radon_h_h2'],
     features_df['radon_h_N1'],
     features_df['radon_h_N2'],
     features_df['radon_h_vocabulary'],
     features_df['radon_h_length'],
     features_df['radon_h_calculated_length'],
     features_df['radon_h_volume'],
     features_df['radon_h_difficulty'],
     features_df['radon_h_effort'],
     features_df['radon_h_time'],
     features_df['radon_h_bugs']) = zip(*radon_h)
    # set is_error flag
    features_df['radon_h_is_error'] = features_df.radon_h_h1.isna()
    # logging results
    n_radon_h_error = features_df.radon_h_is_error.sum()
    n_radon_h_success = sum(features_df.radon_h_h1 >= 0)
    logger.info("Extracted radon halstead metrics: {} scripts, {} successes, {} errors."
                .format(n, n_radon_h_success, n_radon_h_error))
    
    # radon maintainability index metric
    logger.info("Start extracting radon maintainability index metric.")
    # extraction
    radon_mi = [try_radon_mi(scripts_df.content, index, logger) for index in scripts_df.index]
    # save results to features_df
    features_df['radon_mi'] = radon_mi
    #set is_error flag
    features_df['radon_mi_is_error'] = features_df.radon_mi.isna()
    # logging results
    n_radon_mi_error = features_df.radon_mi_is_error.sum()
    n_radon_mi_success = sum(features_df.radon_mi >= 0)
    logger.info("Extracted radon maintainability index metric: {} scripts, {} successes, {} errors."
                .format(n, n_radon_mi_success, n_radon_mi_error))
    
    # export features_df as pickle file to interim folder
    features_df.to_pickle(os.path.join(interim_path, 'features_df.pkl'))
    logger.info("Saved script_df to {}."
                .format(os.path.join(interim_path, 'features_df.pkl')))
    
    # logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to extract the features: {}".format(time_passed))


def try_radon_raw(series, index, logger):
    """Tries to extract radon raw metrics.
    
    Input: pandas series of content, index of row to analyze content
    Output: list of seven metrics: [loc, lloc, sloc, comments, multi, blank]"""
    try:
        result = radon.raw.analyze(series[index])
        logger.debug("Successfully extracted radon raw metrics of file {}."
                     .format(index))
        return list(result)
    except:
        logger.exception("Failed to extract radon raw metrics of file      {}."
                     .format(index))
        return [np.nan]*7


def try_radon_cc(series, index, logger):
    """Tries to extract radon cyclomatic complexity metrics.
    
    Input: pandas series of content, index of row to analyze content
    Output: list of blocks, which are either function, class, or method"""
    try:
        result = radon.complexity.cc_visit(series[index])
        logger.debug("Successfully extracted radon cyclometric complexity metrics of file {}."
                     .format(index))
        return result
    except:
        logger.exception("Failed to extract radon cyclometric complexity metrics of file      {}."
                     .format(index))
        return ''


def try_radon_h(series, index, logger):
    """Tries to extract radon halstead metrics.
    
    Input: pandas series of content, index of row to analyze content
    Output: list of 12 metrics: [h1, h2, N1, N2, vocuabulary, length, 
            calculated_length, volume, difficulty, effort, time, bugs]"""
    try:
        result = radon.metrics.h_visit(series[index])
        logger.debug("Successfully extracted radon halstead metrics of file {}."
                     .format(index))
        return list(result)
    except:
        logger.exception("Failed to extract radon halstead metrics of file      {}."
                     .format(index))
        return [np.nan]*12


def try_radon_mi(series, index, logger):
    """Tries to extract radon maintainability index metric.
    
    Input: pandas series of content, index of row to analyze content
    Output: maintainability index metric"""
    try:
        result = radon.metrics.mi_visit(series[index], True)
        logger.debug("Successfully extracted radon maintainability index metric of file {}."
                     .format(index))
        return result
    except:
        logger.exception("Failed to extract radon maintainability index metric of file      {}."
                     .format(index))
        return np.nan


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