import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV


def fit_score_model(Z, y, slices_train, slices_test, alg='ElasticNet', n_jobs=-1, maxiter=10000, save_model=True):
    """
    train and predict cognitive scores using outputs from normative model
    
    Args:
        Z (np.ndarray): The deviation scores.
        y (np.ndarray): The cognitive scores.
        alg (str): The algorithm to use for fitting the model.
        n_jobs (int): The number of jobs to run in parallel.
        maxiter (int): The maximum number of iterations for the model.
        save_model (bool): Whether to save the model.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The predicted cognitive scores and the deviation scores.
    """
    algs = {'ElasticNet': ElasticNetCV, 'Lasso': LassoCV, 'Ridge': RidgeCV}
    all_predictions = []
    all_betas = []
    for _, (train_idx, test_idx) in enumerate(zip(slices_train, slices_test)):

        Z_train = Z[train_idx]
        Z_test = Z[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = algs[alg](cv=3, random_state=42, n_jobs=n_jobs, max_iter=maxiter)  # Adjust alpha as needed
        model.fit(Z_train, y_train)

        predictions = model.predict(Z_test)
        all_predictions.append(predictions)
        
    if save_model:
        with open(f'{alg}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    # TODO: reshape all_predictions to match the shape of y
    # TODO: save betas for future predictions
    return all_predictions

def predict_score(Z, coefs_filename):
    """A function that loads the pre-trained coefficients and predicts the scores for new deviation scores"""
    with open(coefs_filename, 'rb') as f:
        coefs = pickle.load(f)
    return np.dot(Z, coefs)
