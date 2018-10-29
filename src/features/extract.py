# -*- coding: utf-8 -*-

import os, logging, argparse, re
import pandas as pd
import numpy as np
from time import time
import collections
import radon.raw, radon.complexity, radon.metrics
import subprocess
from collections import Counter

#%% define main function which performs whole extraction
def main(interim_path = "data/interim"):
    
    """Extracts all features from the source codes.
    
    Output extracted_df includes following source code metrics:
        Radon:
        - Raw metrics
        - Cyclomatic Complexity metrics
        - Halstead Metrics
        - Maintainability Index metric
        Pylint:
        - Statistics by type: # of module, class, method, function
        - Raw metrics: # of code, docstring, comment, empty lines
        - Duplication: # of duplicated lines
        - Messages by category: # of convention, refactor, warning, error 
          messages by pylint
        - Messages: # of all occurrences by specific message id
        Manually:
        - a column 'uses_module_<module>' for each module used
    Every metric type has a flag column <metric type>_is_error indicating
    if an error occurred while trying to extract the respective metrics."""
    
    # logging
    logger = logging.getLogger(__name__)
    
    # normalize path
    interim_path = os.path.normpath(interim_path)
    logger.debug("Path to iterim data normalized: {}".format(interim_path))
    
    # load tabled_df and create copy for storing features
    tabled_df = pd.read_pickle(os.path.join(interim_path, 'tabled_df.pkl'))
    extracted_df = tabled_df[['repo_id']].copy()
    logger.info("Created extracted_df.")

    # start extracting source code metrics
    start = time()
    n = len(extracted_df)
    
    #%% radon raw metrics
    logger.info("Start extracting radon raw metrics.")
    # extraction
    radon_raw = [try_radon_raw(tabled_df.content, index, logger) 
                for index in tabled_df.index]
    # save results to extracted_df
    (extracted_df['radon_raw_loc'],
     extracted_df['radon_raw_lloc'],
     extracted_df['radon_raw_sloc'],
     extracted_df['radon_raw_comments'],
     extracted_df['radon_raw_multi'],
     extracted_df['radon_raw_blank'],
     extracted_df['radon_raw_single_comments']) = zip(*radon_raw)
    # set is_error flag
    extracted_df['radon_raw_is_error'] = extracted_df.radon_raw_loc.isna()
    # logging results
    n_radon_raw_error = extracted_df.radon_raw_is_error.sum()
    n_radon_raw_success = sum(extracted_df.radon_raw_loc >= 0)
    logger.info("Finished extracting radon raw metrics.")
    logger.info("{} scripts, {} successes, {} errors"
                .format(n, n_radon_raw_success, n_radon_raw_error))
    
    #%% radon cyclomatic complexity metrics
    logger.info("Start extracting radon cyclomatic complexity metrics.")
    # extraction
    radon_cc = [try_radon_cc(tabled_df.content, index, logger)
                for index in tabled_df.index]
    # save results to extracted_df
    extracted_df['radon_cc_mean'] = list(
                        map(radon.complexity.average_complexity, radon_cc))
    extracted_df['radon_cc_sum'] = [sum([obj.complexity for obj in blocks])
                                    for blocks in radon_cc]
    # set is_error flag
    extracted_df['radon_cc_is_error'] = [isinstance(x, str) for x in radon_cc]
    # logging results
    n_radon_cc_error = sum(extracted_df.radon_cc_is_error)
    n_radon_cc_success = sum([isinstance(x, list) for x in radon_cc])
    logger.info("Finished extracting radon cyclomatic complexity metrics.")
    logger.info("{} scripts, {} successes, {} errors"
                .format(n, n_radon_cc_success, n_radon_cc_error))
    
    #%% radon halstead metrics
    logger.info("Start extracting radon halstead metrics.")
    # extraction
    radon_h = [try_radon_h(tabled_df.content, index, logger)
                for index in tabled_df.index]
    # save results to extracted_df
    (extracted_df['radon_h_h1'],
     extracted_df['radon_h_h2'],
     extracted_df['radon_h_N1'],
     extracted_df['radon_h_N2'],
     extracted_df['radon_h_vocabulary'],
     extracted_df['radon_h_length'],
     extracted_df['radon_h_calculated_length'],
     extracted_df['radon_h_volume'],
     extracted_df['radon_h_difficulty'],
     extracted_df['radon_h_effort'],
     extracted_df['radon_h_time'],
     extracted_df['radon_h_bugs']) = zip(*radon_h)
    # set is_error flag
    extracted_df['radon_h_is_error'] = extracted_df.radon_h_h1.isna()
    # logging results
    n_radon_h_error = extracted_df.radon_h_is_error.sum()
    n_radon_h_success = sum(extracted_df.radon_h_h1 >= 0)
    logger.info("Finished extracting radon halstead metrics.")
    logger.info("{} scripts, {} successes, {} errors."
                .format(n, n_radon_h_success, n_radon_h_error))
    
    #%% radon maintainability index metric
    logger.info("Start extracting radon maintainability index metric.")
    # extraction
    radon_mi = [try_radon_mi(tabled_df.content, index, logger)
                for index in tabled_df.index]
    # save results to extracted_df
    extracted_df['radon_mi'] = radon_mi
    #set is_error flag
    extracted_df['radon_mi_is_error'] = extracted_df.radon_mi.isna()
    # logging results
    n_radon_mi_error = extracted_df.radon_mi_is_error.sum()
    n_radon_mi_success = sum(extracted_df.radon_mi >= 0)
    logger.info("Finished extracting radon maintainability index metric.")
    logger.info("{} scripts, {} successes, {} errors."
                .format(n, n_radon_mi_success, n_radon_mi_error))
    
    #%% pylint
    logger.info("Start extracting pylint metrics.")
    # extraction
    counters = [try_pylint(tabled_df, index, logger)
                for index in tabled_df.index]
    pylint = pd.DataFrame(counters)
    # drop 'percent duplicated lines', already have 'nb duplicated lines'
    pylint.drop(columns='percent duplicated lines', inplace=True)
    # align column naming
    pylint.rename(lambda x: "pylint_" + x.replace('-', '_').replace(' ', '_'),
                axis = 1, inplace=True)
    # save results to extracted_df
    extracted_df = pd.concat([extracted_df, pylint], axis=1)
    #set is_error flag
    extracted_df['pylint_is_error'] = [counter == collections.Counter()
                                        for counter in counters]
    # logging results
    n_pylint_error = extracted_df.pylint_is_error.sum()
    n_pylint_success = n-n_pylint_error
    logger.info("Finished extracting pylint metrics.")
    logger.info("{} scripts, {} successes, {} errors."
                .format(n, n_pylint_success, n_pylint_error))
    
    #%% extract used modules
    
    # list of modules per repo
    modules = []
    for index, repo in tabled_df.iterrows():
        directs = re.findall(r"^[\s]*import ([a-zA-Z0-9]+)",
                             repo.content, re.MULTILINE)
        froms = re.findall(r"from ([a-zA-Z0-9]+)[a-zA-Z0-9.]* import",
                           repo.content)
        together = Counter(set(directs) | set(froms))
        modules.append(together)
        logger.debug("Script {} uses following modules: {}"
                     .format(index, list(together.keys())))
    
    # total counts
    total = Counter()
    for m in modules:
        total.update(m)
    logger.info("In total {} different modules are used."
                .format(len(total)))
    logger.info("Top 10 packages:\n{}"
                .format(total.most_common(10)))
    
    # create df of one-hot-encoded module features
    module_names = sorted(list(total.keys()), key=str.lower)
    dictionary = {}
    for index in tabled_df.index:
        dictionary[index] = [modules[index][name] for name in module_names]
    module_names = ['uses_module_'+x for x in module_names]
    df = pd.DataFrame.from_dict(dictionary,
                                orient='index',
                                columns=module_names)
    
    # concatenate to extracted_df
    extracted_df = pd.concat([extracted_df, df], axis=1)
    
    #%% export extracted_df as pickle file to interim folder
    extracted_df.to_pickle(os.path.join(interim_path, 'extracted_df.pkl'))
    logger.info("Saved script_df to {}."
                .format(os.path.join(interim_path, 'extracted_df.pkl')))
    
    #%% logging time passed
    end = time()
    time_passed = pd.Timedelta(seconds=end-start).round(freq='s')
    logger.info("Time needed to extract the features: {}".format(time_passed))

#%% define some helper functions which tries to extract the code metrics

def try_radon_raw(series, index, logger):
    """Tries to extract radon raw metrics.
    
    Input: pandas series of content, index of row to analyze content
    Output: list of 7 metrics: [loc, lloc, sloc, comments, multi, blank]"""
    try:
        result = radon.raw.analyze(series[index])
        logger.debug("Successfully extracted radon_raw metrics of file {}."
                     .format(index))
        return list(result)
    except Exception:
        logger.exception("Failed to extract radon_raw metrics of file      {}."
                     .format(index))
        return [np.nan]*7

def try_radon_cc(series, index, logger):
    """Tries to extract radon cyclomatic complexity metrics.
    
    Input: pandas series of content, index of row to analyze content
    Output: list of blocks, which are either function, class, or method"""
    try:
        result = radon.complexity.cc_visit(series[index])
        logger.debug("Successfully extracted radon_cc metrics of file {}."
                     .format(index))
        return result
    except Exception:
        logger.exception("Failed to extract radon_cc metrics of file      {}."
                         .format(index))
        return ''

def try_radon_h(series, index, logger):
    """Tries to extract radon halstead metrics.
    
    Input: pandas series of content, index of row to analyze content
    Output: list of 12 metrics: [h1, h2, N1, N2, vocuabulary, length, 
            calculated_length, volume, difficulty, effort, time, bugs]"""
    try:
        result = radon.metrics.h_visit(series[index])
        logger.debug("Successfully extracted radon_h metrics of file {}."
                     .format(index))
        return list(result)
    except Exception:
        logger.exception("Failed to extract radon_h metrics of file      {}."
                         .format(index))
        return [np.nan]*12

def try_radon_mi(series, index, logger):
    """Tries to extract radon maintainability index metric.
    
    Input: pandas series of content, index of row to analyze content
    Output: maintainability index metric"""
    try:
        result = radon.metrics.mi_visit(series[index], True)
        logger.debug("Successfully extracted radon_mi metric of file {}."
                     .format(index))
        return result
    except Exception:
        logger.exception("Failed to extract radon_mi metric of file      {}."
                         .format(index))
        return np.nan

def try_pylint(df, index, logger):
    """Tries to extract pylint metrics.
    
    Input: pandas DataFrame of content, index of row to analyze content
    Output: Counter object from collections package with pylint metrics"""
    args = ['pylint', df.name[index], '--reports=y']
    cwd = df.path[index]
    # tries to perform pylint on respective file
    try:
        stdout = subprocess.check_output(args, cwd=cwd).decode('utf-8')
        logger.debug("Successfully extracted pylint metrics of file {}."
                     .format(index))
    # may result in non-zero return code, see:
    # https://stackoverflow.com/questions/49100806/
    # pylint-and-subprocess-run-returning-exit-status-28
    except subprocess.CalledProcessError as e:
        stdout = e.output.decode('utf-8')
        logger.debug("Successfully extracted pylint metrics of file {}."
                     .format(index))
        logger.debug("(with non-zero exit status: {})."
                     .format(e.returncode))
    except Exception:
        logger.exception("Failed to extract pylint metrics of file      {}."
                     .format(index))
        return collections.Counter()
    # define pattern for pylint output: "|<name>    |<count>      |"
    pattern = "^\|(?P<name>[a-z][-\sa-z]+[a-z])\s*\|(?P<count>\d+)\s*\|"
    # match pattern and extracts list of 2-tuples: (<name>, <count>)
    matches = re.findall(pattern, stdout, flags=re.MULTILINE)
    # convert list of matches to Counter object
    counter = collections.Counter({name:int(count) for name, count in matches})
    return counter

#%%
if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/extract.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Extracts all features from the source codes.")
    parser.add_argument(
            '--interim_path',
            default = "data/interim",
            help = "path to store the output extracted_df.pkl \
                    (default: data/interim)")
    args = parser.parse_args()
    
    # run main
    main(args.interim_path)