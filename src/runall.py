import sys

# forces import statement to also search in cwd (should be .../meta-kaggle)
# see: chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html#more-on-syspath
if '' not in sys.path:
    sys.path.insert(0, '')

from src.data import download

#%% download Github repositories from Team.csv to data/external/repositories

download.main()