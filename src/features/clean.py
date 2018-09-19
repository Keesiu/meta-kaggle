# -*- coding: utf-8 -*-

import os, logging, argparse
import pandas as pd
import collections
from time import time


def main():
    
    """Cleans aggregated data."""
    
    

if __name__ == '__main__':
    
    # configure logging
    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        filename = "logs/aggregate.log",
        datefmt = "%a, %d %b %Y %H:%M:%S")