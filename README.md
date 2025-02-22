# brain-norm
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20-blue)](https://github.com/alawryaguila/multi-view-ae)

Brainnorm is a toolbox that uses normative modeling to predict cognitive scores. This toolbox allows you to fit normative models for your chosen brain measures, and predict cognitive scores for your chosen tasks.

## Why normative modeling?

TBC

## Installation

We recommend creating a new environment to use this toolbox. To create a new environment, you can use the following command:

```bash
conda create -n brainnorm python=3.10
conda activate brainnorm
```

After creating the environment, you can install the toolbox using the following command:

```bash
git clone https://github.com/alawryaguila/brain-norm.git
cd brain-norm
pip install .
```

## Usage

### Prerequisites

You need to have the following data:

- Brain measures and covariates: You need to have dataframes with the brain measures and covariates (e.g. age, sex, etc.).
- Cognitive scores: This is the outcome variable you want to predict, e.g., IQ, mental health scores, learning ability, etc.
- Batch effect: If you want to control for site effects, you need to have a dataframe with the site/batch effect data.

### Fitting the model

Below is a minimal example of how to fit normative models using this toolbox. First, read the data and split the data into training and testing sets.

```python
import pandas as pd
from brainnorm.utils import split_dataframe

# load brain measures and covariates - these dataframes should have the same number of rows
measures = pd.read_csv('measures.csv')
covariates = pd.read_csv('covariates.csv')

# Split the data into training and testing sets
train_idx, test_idx = split_dataframe(measures, folds=3, id_col='subject_id')
```

Here `id_col` is the column that contains the subject IDs, and `folds` is the number of folds to split the data into. This function takes care of repeated scans for the same subject, so that the same subject is not split into different folds, avoiding data leakage.

Next, fit normative models across different folds:

```python
fit_normative_model(
    cov=covariates, resp=measures, 
    train_idx=train_idx, test_idx=test_idx, 
    output_path='output', log_path='log', outputsuffix='', 
    alg: str = 'gpr', optimizer: str = 'powell', save_results: bool = True, 
    batch: Union[pd.DataFrame, np.ndarray, None] = None,
    savemodel: bool = True
)
```
