# meta-kaggle
Analysis of Kaggle Meta Data 2016 for my Master's Thesis "*Artificial Intelligence from a Resource-Based View: A Quantitative Analysis of Machine Learning Source Code*" within my Master in Management at Technical University Munich (TUM) in 2018.

## Motivation

## Getting started

### Clone repository
In your console, navigate to a folder where you want to save the repository. Clone the whole "meta-kaggle" repository with `git clone https://github.com/Keesiu/meta-kaggle.git`.

### Repository structure
The cloned repository has the following basic structure:
```
├───data
│   ├───external
│   │   ├───repositories
│   │   └───repositories_2to3
│   ├───interim
│   ├───processed
│   └───raw
│       ├───kaggle-survey-2017
│       └───meta-kaggle-2016
├───logs
├───models
└───src
    ├───data
    │   └───__pycache__
    ├───features
    │   └───__pycache__
    ├───models
    │   └───__pycache__
    ├───visualization
    └───__pycache__
```
### Create a suitable environment
In order to create a suitable environment with the correct dependencies, use the yaml-file `environment.yaml`. I recommend using Anaconda as an envrionment manager. There, simply use `conda env create -f environment.yaml` to copy my setup (see: https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). Afterwards, activate it with `activate meta-kaggle`.

## Reproduce the results
Follow these steps to reproduce the results. Execute the respective scripts from the console.
IMPORTANT: Stay in top-level folder "*meta-kaggle*", since most scripts need this working directory as a reference path in order to find other files properly.
As an option, you can execute Step #1 - Step #11 alltogether by executing runall.py: `python src/runall.py`.

### Optional: Reset whole repository to start from downloading all files.
#### Step #0: `python src/reset.py`
Deletes all downloaded repositories, log-files, pickled interim and processed data, models and visualizations. 

### Data Collection and Preparation
#### Step #1: `python src/data/download.py`
Downloads all publicly available repositories from meta kaggle 2016.
#### Step #2: `python src/data/reduce.py`
Reduces all downloaded data by deleting all non-Python files.
#### Step #3: `python src/data/translate2to3.py`
Translates all Python scripts written in 2.X to 3.X, in order for the source code analysis tools "Radon" and "Pylint" to work fine.
#### Step #4: `python src/data/table.py`
Tables all scripts and their respective code.

### Feature Engineering and Selection
#### Step #5: `python src/features/extract.py`
Extracts all features from the code.
#### Step #6: `python src/features/aggregate.py`
Aggregates all features from script-level to repository-level.
#### Step #7: `python src/features/clean.py`
Cleans the data.
#### Step #8: `python src/features/select.py`
Select desired features for modeling, based on theory and hypotheses.
#### Step #9: `python src/features/pca.py`
Optional: Performs PCA on data and show structure of main components.

### Modelling
#### Step #10: `python src/models/train.py`
Trains the models.

### Visualization
#### Step #11: `python src/visualization/visualize.py`
Produces some visualizations.

## Results

## Further readings
