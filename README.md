# brain-norm
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20-blue)](https://github.com/alawryaguila/multi-view-ae)

Brainnorm is a toolbox that uses normative modeling to predict cognitive scores. This toolbox allows you to fit normative models for your chosen brain measures, and predict cognitive scores for your chosen tasks.

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

