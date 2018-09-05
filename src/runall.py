import os
from src.data import download

download.main("data/raw/meta-kaggle-2016/Teams.csv", "data/external/repositories")
