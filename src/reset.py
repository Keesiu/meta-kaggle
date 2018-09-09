import os, shutil, argparse


def main(repo_path = "data/external/repositories", logs_path = "logs"):
    
    # normalize path
    repo_path = os.path.normpath(repo_path)
    
    # delete all downloaded repositories
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    
    # delete all log-files
    for file in os.listdir(logs_path):
        if file[-4:] == '.log':
            os.remove(os.path.join(logs_path, file))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description="Resets project for clean re-run.")
    parser.add_argument(
            '-r', '--repo_path',
            default="data/external/repositories",
            help="path to downloaded repositories (default: data/external/repositories)")
    parser.add_argument(
            '-l', '--logs_path',
            default="logs",
            help="path to log-files (default: logs)")
    args = parser.parse_args()
    
    main(args.repo_path, args.logs_path)