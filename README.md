# brain-norm
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20-blue)](https://github.com/alawryaguila/multi-view-ae)

Brainnorm is a toolbox that uses normative modeling to predict cognitive scores. This toolbox allows you to fit normative models for your chosen brain measures, and predict cognitive scores for your chosen tasks.

## Why normative modeling?

For decades, pediatricians have used normative modeling for developing Growth Charts. Similarly, for predicting brain development, normative modeling is a critical tool for predicting the underlying framework. With this neuroscience package, a variety of normative modeling algorithms are offered including Hierarchical Bayesian Modeling, Gaussian Processing Regression, etc.

## Installation

We recommend creating a new environment to use this toolbox. To create a new environment, you can use the following command:

```bash
conda create -n brainnorm python=3.10
conda activate brainnorm
```

After creating the environment, you can install the toolbox using the following command:

```bash
git clone https://github.com/elainez1006/brain-norm.git
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
from brainnorm.utils import read_dataframe, split_dataframe

# load brain measures and covariates - these dataframes should have the same number of rows (i.e., subjects are matched)
measures = pd.read_csv('measures.csv')
covariates = pd.read_csv('covariates.csv')

# Split the data into training and testing sets
train_idx, test_idx = split_dataframe(measures, folds=3, id_col='subject_id')
```

Here `id_col` is the column that contains the subject IDs, and `folds` is the number of folds to split the data into. This function takes care of repeated scans for the same subject, so that the same subject is not split into different folds, avoiding data leakage.

Next, fit normative models across different folds, using the indices generated in the previous step:

```python
fit_normative_model(
    covariates=covariates, response=measures, 
    train_idx=train_idx, test_idx=test_idx, 
    output_path='output', log_path='log', outputsuffix='', 
    alg='gpr', optimizer='powell', save_results=True, 
    savemodel=True
)
```

This command will fit normative models across different folds, and save the results in the `output` directory. You can use different algorithms and optimizers to fit the normative models. Available options of `alg` include `gpr`, `blr`, `hbr`, etc. More details can be found in the documentation.

To save results of the normative models, you can set `save_results=True`. This will save the results in a csv file with the provided `outputsuffix`.

To save the model, you can set `savemodel=True`. This will save the model in a pickle file with the provided `outputsuffix`.

### Train and predict cognitive scores

After fitting the normative models, you can train and predict cognitive scores using the outputs from the normative models. This is done using the `fit_score_model` function.

First, load the deviation scores (output from the normative models) and the cognitive scores:

```python
# load deviation scores
with open(f'Z_outputsuffix.pkl', 'rb') as f:
    Z = pickle.load(f)

# load the dataframe containing the cognitive scores
# note that this dataframe should have rows matching measures and covariates
y = read_dataframe('cognitive_scores.csv')
```

Then, train and predict cognitive scores:

```python
predictions = fit_score_model(
    Z=Z, y=y, slices_train=train_idx, slices_test=test_idx, 
    alg='ElasticNet', n_jobs=-1, maxiter=10000, save_model=True
)
```

Available options of `alg` include `ElasticNet`, `Lasso`, `Ridge`, etc. For more details on the available algorithms, please refer to this page [Lasso vs Ridge vs Elastic Net | ML](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/). Here we recommend using `ElasticNet` as it is a good balance between sparsity and stability. `n_jobs` is the number of jobs to run in parallel, and `maxiter` is the maximum number of iterations for the model.

The returned `predictions` contains the predicted cognitive scores for the testing set in each fold.

To save the model, you can set `save_model=True`. This will save the prediction coefficients for future predictions.

### Predict deviation scores and cognitive scores using pre-trained models

This toolbox allows you to save the pre-trained models to predict deviation scores and thus cognitive scores for new datasets. Here we provide an example of how to do this.

First, predict deviation scores using the pre-trained models:

```python
# load a new dataset. Note that this new dataset should have the same columns as the training data.
new_measures = pd.read_csv('new_measures.csv')
new_covariates = pd.read_csv('new_covariates.csv')

# predict deviation scores
Z, yhat, s2 = predict_normative_model(
    covariates=new_covariates, measures=new_measures, 
    model_path='pretrained_model.pkl', alg='gpr'
)
```

Next, predict cognitive scores using the deviation scores:

```python
# predict cognitive scores
predictions = predict_score(Z=Z, coefs_filename='coefs_pretrained_model.pkl')
```

The returned `predictions` contains the predicted cognitive scores for the new dataset.
