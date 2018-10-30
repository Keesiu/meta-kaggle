# meta-kaggle
Analysis of Kaggle Meta Data 2016 for my Master's Thesis "*Artificial Intelligence from a Resource-Based View: A Quantitative Analysis of Machine Learning Source Code*" within my Master in Management at Technical University Munich (TUM) in 2018.



## Getting started

### Clone repository
In your console, navigate to a folder where you want to save the repository. Clone the whole "meta-kaggle" repository with `git clone https://github.com/Keesiu/meta-kaggle.git`.

### Repository structure
The cloned repository has the following basic structure:
```
└───meta-kaggle                             <- IMPORTANT: must be current working directory
    │   .gitignore                          <- gitignore-file in order to only upload small data
    │   environment.yaml                    <- yaml-file to reproduce suitable environment
    │   LICENSE
    │   README.md                           <- markdown-file on how to use this repository
    │
    ├───data                                <- where all data is stored
    │   ├───external                        <- data from third party sources
    │   │   │   .gitkeep                    <- makes Github to keep this empty folder
    │   │   │
    │   │   ├───repositories                <- downloaded external repositories
    │   │   │   ├───<Id_1>                  <- respective Id from Teams.csv
    │   │   │   │   └───<Repository_name_1> <- name of downloaded repository
    │   │   │   │       └───...             <- individual content of this repository
    │   │   │   │
    │   │   │   ├───<Id_2>
    │   │   │   │   └───<Repository_name_2>
    │   │   │   │       └───...
    │   │   │   │
    │   │   │   ...
    │   │   │
    │   │   └───repositories_2to3           <- translated external repositories
    │   │       ├───<Id_1>
    │   │       │   └───<Repository_name_1>
    │   │       │       └───...
    │   │       │
    │   │       ├───<Id_2>
    │   │       │   └───<Repository_name_2>
    │   │       │       └───...
    │   │       │
    │   │       ...
    │   │
    │   ├───interim                         <- intermediate data that has been transformed
    │   │       .gitkeep
    │   │
    │   ├───processed                       <- the final data sets for feature selection and modeling
    │   │       .gitkeep
    │   │
    │   └───raw                             <- The original, immutable raw data
    │       │   .gitkeep
    │       │
    │       └───meta-kaggle-2016            <- IMPORTANT: needs to be inserted manually
    │               CompetitionHostSegments.csv
    │               Competitions.csv
    │               Datasets.csv
    │               DatasetVersions.csv
    │               EvaluationAlgorithms.csv
    │               ForumMessages.csv
    │               Forums.csv
    │               ForumTopics.csv
    │               hashes.txt
    │               RewardTypes.csv
    │               ScriptLanguages.csv
    │               ScriptProjects.csv
    │               ScriptRunOutputFileExtensions.csv
    │               ScriptRunOutputFileGroups.csv
    │               ScriptRunOutputFiles.csv
    │               ScriptRuns.csv
    │               Scripts.csv
    │               ScriptVersions.csv
    │               ScriptVotes.csv
    │               Sites.csv
    │               Submissions.csv
    │               TeamMemberships.csv
    │               Teams.csv               <- most important file, includes Ranking and GithubRepolink
    │               Users.csv
    │               ValidationSets.csv
    │
    ├───logs                                <- log-files of all scripts that run
    │       .gitkeep
    │
    ├───models                              <- trained and serialized models and model summaries
    │       .gitkeep
    │
    └───src                                 <- Source code to use in this project
        │   reset.py
        │   runall.py
        │   __init__.py                     <- makes src a Python module
        │
        ├───data                            <- scripts for data collection and preparation
        │       download.py
        │       reduce.py
        │       table.py
        │       translate2to3.py
        │       __init__.py
        │
        ├───features                        <- scripts for feature engineering and selection
        │       aggregate.py
        │       clean.py
        │       extract.py
        │       select.py
        │       __init__.py
        │
        ├───models                          <- scripts to train models
        │       train.py
        │       __init__.py
        │
        └───visualization                   <- scripts to create visualizations
                visualize.py
                __init__.py
```

### Include raw data "meta-kaggle-2016"
The raw data is essentially the kaggle meta data from 2016. Its data set "meta-kaggle-2016" needs to be placed into the folder `data/raw'.

> **NOTE**: This kaggle dataset is not available on Kaggle's webpage anymore, since a newer version (Meta Kaggle 2.0) was published (see: https://www.kaggle.com/kaggle/meta-kaggle/data).

### Create a suitable environment
In order to create a suitable environment with the correct dependencies, use the yaml-file `environment.yaml`. I recommend using Anaconda as an environment manager. There, simply use `conda env create -f environment.yaml` to copy my setup (see: https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Afterwards, activate it with `activate meta-kaggle`.



## Reproduce the results

Follow these steps to reproduce the results. Execute the respective scripts from the console.

> **IMPORTANT**: Stay in top-level folder `meta-kaggle`, since most scripts need this working directory as a reference path in order to find other files properly.

> **NOTE:** You can execute Step #1 - Step #10 alltogether by executing runall.py: `python src/runall.py`.

### OPTIONAL: Reset whole repository
#### Step #0: `python src/reset.py`
Deletes all downloaded repositories, log-files, pickled interim and processed data, models and visualizations for a clean start from the very beginning. Note, that you especially need to download all external repositories again afterwards.

### SHORT WAY: Run everything automatically
#### Step #1-10: `python src/runall.py`
Runs all the steps described in the following. Automatically skips unneccessary steps.

### LONG WAY: Run every script manually step-by-step

### Data Collection and Preparation
#### Step #1: `python src/data/download.py`
Downloads all publicly available repositories from meta kaggle 2016.
#### Step #2: `python src/data/reduce.py`
Reduces all downloaded data by deleting all non-Python files.
#### Step #3: `python src/data/translate2to3.py`
Translates all Python scripts written in 2.X to 3.X, in order for the source code analysis tools "Radon" and "Pylint" to work.
#### Step #4: `python src/data/table.py`
Tables all scripts and their respective code content.

### Feature Engineering and Selection
#### Step #5: `python src/features/extract.py`
Extracts all features from the code.
#### Step #6: `python src/features/aggregate.py`
Aggregates all features from script-level to repository-level.
#### Step #7: `python src/features/clean.py`
Cleans the data by final feature engineering and outlier removement.
#### Step #8: `python src/features/select.py`
Select desired features for modeling, based on theory and hypotheses.

### Modelling
#### Step #9: `python src/models/train.py`
Trains the models.

### Visualization
#### Step #10: `python src/visualization/visualize.py`
Produces some visualizations.
