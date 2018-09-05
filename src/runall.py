from src.data import download

# download Github repositories from Team.csv to data/external/repositories
download.main("data/raw/meta-kaggle-2016/Teams.csv", "data/external/repositories")