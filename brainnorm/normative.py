import numpy as np
import pandas as pd
from pathlib import Path
import pcntoolkit as pcn
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
        data (pd.DataFrame): The data to fit the model to.
        slices_train (List[int]): Indices of the training set.
        slices_test (List[int]): Indices of the testing set.
        
    Returns:
        pd.DataFrame: The fitted model.
    """
    results = []
    for fold_idx, (slices_train, slices_test) in enumerate(zip(train_idx, test_idx)):
        cov_train = cov.iloc[slices_train]
        cov_test = cov.iloc[slices_test]
        resp_train = resp.iloc[slices_test]
        resp_test = resp.iloc[slices_test]
        with open(f'cov_train_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(cov_train, f)
        with open(f'resp_train_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(resp_train, f)
        with open(f'cov_test_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(cov_test, f)
        with open(f'resp_test_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(resp_test, f)
        results = pcn.normative.estimate(
            covfile=f'cov_train_fold{fold_idx}.pkl', 
            respfile=f'mri_train_fold{fold_idx}.pkl', 
            testcov=f'cov_test_fold{fold_idx}.pkl', 
            testresp=f'mri_test_fold{fold_idx}.pkl', 
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
        results.append(dev_scores)
    return results


