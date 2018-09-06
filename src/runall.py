import sys

# forces import statement to also search in cwd (should be .../meta-kaggle)
print(sys.path)
if '' not in sys.path:
    sys.path.insert(0, '')
print(sys.path)

from src.data import download

#%% download Github repositories from Team.csv to data/external/repositories
# download.main()
