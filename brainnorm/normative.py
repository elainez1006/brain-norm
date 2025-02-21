import numpy as np
import pandas as pd
from pathlib import Path
from pcntoolkit.normative import estimate, predict
from typing import List
import pickle


def fit_normative_model(cov: pd.DataFrame, resp: pd.DataFrame, 
                        train_idx: List[List[int]], test_idx: List[List[int]], 
                        output_path: str, log_path: str, outputsuffix: str, 
                        alg: str = 'gpr', optimizer: str = 'powell', save_results: bool = True, 
                        savemodel: bool = True):
    """
    Fit normative models to the data.
    
    Args:
        cov (pd.DataFrame): The covariance data.
        resp (pd.DataFrame): The response data.
        train_idx (List[List[int]]): Indices of the training set.
        test_idx (List[List[int]]): Indices of the testing set.
        output_path (str): The path to the output directory.
        log_path (str): The path to the log directory.
        outputsuffix (str): The suffix to add to the output files.
        alg (str): The algorithm to use for fitting the model.
        optimizer (str): The optimizer to use for fitting the model.
        save_results (bool): Whether to save the results.
        savemodel (bool): Whether to save the model.
        
    Returns:
        pd.DataFrame: The fitted model.
    """
    all_results = []
    for fold_idx, (slices_train, slices_test) in enumerate(zip(train_idx, test_idx)):
        cov_train = cov.iloc[slices_train]
        cov_test = cov.iloc[slices_test]
        resp_train = resp.iloc[slices_train]
        resp_test = resp.iloc[slices_test]
        with open(f'cov_train_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(cov_train, f)
        with open(f'resp_train_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(resp_train, f)
        with open(f'cov_test_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(cov_test, f)
        with open(f'resp_test_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(resp_test, f)
        estimate(
            covfile=f'cov_train_fold{fold_idx}.pkl', 
            respfile=f'resp_train_fold{fold_idx}.pkl', 
            testcov=f'cov_test_fold{fold_idx}.pkl', 
            testresp=f'resp_test_fold{fold_idx}.pkl', 
            inscaler='standardize', 
            outscaler='standardize', 
            output_path=output_path, 
            log_path=log_path, 
            outputsuffix=f'{outputsuffix}_fold{fold_idx}', 
            alg=alg, 
            optimizer=optimizer, 
            save_results=save_results,
            savemodel=savemodel
        )
        # load the deviation scores
        with open(f'Z_{outputsuffix}_fold{fold_idx}.pkl', 'rb') as f:
            dev_scores = pickle.load(f)
        all_results.append(dev_scores)
    return all_results

def predict_normative_model(model_path: str, test_data: pd.DataFrame, alg: str = 'gpr'):
    """
    Make predictions for unseen subjects using a normative model.
    
    Args:
        model_path (str): The path to the normative model.
        test_data (pd.DataFrame): The data to make predictions for. (Must match the training data)
        alg (str): The algorithm to use for prediction.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The predicted mean, the variance, and the deviation scores.
    """
    cov_file = 'test_data.pkl'
    with open(cov_file, 'wb') as f:
        pickle.dump(test_data, f)
    yhat, s2, Z = predict(cov_file, alg=alg, outscaler='standardize', model_path=model_path)
    return yhat, s2, Z
