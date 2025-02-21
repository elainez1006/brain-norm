import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def train_and_predict_cognitive_scores(Z, y, slices_train, slices_test, alg='ElasticNet', n_jobs=-1, maxiter=10000):
    """
    train and predict cognitive scores using outputs from normative model
    
    Args:
        Z (np.ndarray): The deviation scores.
        y (np.ndarray): The cognitive scores.
        alg (str): The algorithm to use for fitting the model.
        n_jobs (int): The number of jobs to run in parallel.
        maxiter (int): The maximum number of iterations for the model.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The predicted cognitive scores and the deviation scores.
    """
    algs = {'ElasticNet': ElasticNetCV, 'Lasso': LassoCV, 'Ridge': RidgeCV}
    scaler = StandardScaler()
    all_predictions = []
    for _, (train_idx, test_idx) in enumerate(zip(slices_train, slices_test)):

        Z_train = scaler.fit_transform(Z[train_idx])
        Z_test = scaler.transform(Z[test_idx])

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = algs[alg](cv=3, random_state=42, n_jobs=n_jobs, max_iter=maxiter)  # Adjust alpha as needed
        model.fit(Z_train, y_train)

        predictions = model.predict(Z_test)
        all_predictions.append(predictions)
    return all_predictions

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))