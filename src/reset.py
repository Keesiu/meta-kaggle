# -*- coding: utf-8 -*-

import os, shutil, argparse


def main(repos_path, interim_path, processed_path, logs_path, models_path):
    
    """Resets project for clean re-run."""
    
    # normalize path
    repos_path = os.path.normpath(repos_path)
    
    # delete all downloaded repositories
    if os.path.exists(repos_path):
        shutil.rmtree(repos_path)
    
    # delete all 2to3-transformed repositories
    if os.path.exists(repos_path + '_2to3'):
        shutil.rmtree(repos_path + '_2to3')
    
    # delete all pickle files in interim data
    for file in os.listdir(interim_path):
        if file[-4:] == '.pkl':
            os.remove(os.path.join(interim_path, file))
    
    # delete all pickle files in processed data
    for file in os.listdir(processed_path):
        if file[-4:] == '.pkl':
            os.remove(os.path.join(processed_path, file))
    
    # delete all log-files in logs
    for file in os.listdir(logs_path):
        if file[-4:] == '.log':
            os.remove(os.path.join(logs_path, file))
    
    # delete all pkl- and txt-files in models
    for file in os.listdir(models_path):
        if file[-4:] == '.pkl' or file[-4:] == '.txt':
            os.remove(os.path.join(models_path, file))

if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser(
            description = "Resets project for clean re-run.")
    parser.add_argument(
            '--repos_path',
            default = "data/external/repositories",
            help = "path to downloaded repositories \
                    (default: data/external/repositories)")
    parser.add_argument(
            '--interim_path',
            default = "data/interim",
            help = "path to interim data (default: data/interim)")
    parser.add_argument(
            '--processed_path',
            default = "data/processed",
            help = "path to processed data (default: data/processed)")
    parser.add_argument(
            '--logs_path',
            default = "logs",
            help = "path to log files (default: logs)")
    parser.add_argument(
            '--models_path',
            default = "models",
            help = "path to the trained models (default: models)")
    args = parser.parse_args()
    
    # run main
    main(args.repos_path,
         args.interim_path,
         args.processed_path,
         args.logs_path,
         args.models_path)