import numpy as np
import pandas as pd
from pathlib import Path
from pcntoolkit.normative import estimate, predict
from typing import List, Union
import pickle
from brainnorm.utils import reorder_data


def fit_normative_model(covariates: Union[pd.DataFrame, np.ndarray], response: Union[pd.DataFrame, np.ndarray], 
                        train_idx: List[List[int]], test_idx: List[List[int]], 
                        output_path: str, log_path: str, outputsuffix: str, 
                        alg: str = 'gpr', optimizer: str = 'powell', save_results: bool = True, 
                        batch: Union[pd.DataFrame, np.ndarray, None] = None,
                        save_model: bool = True):
    """
    Fit normative models to dataframe or numpy array.
    
    Args:
        covariates (Union[pd.DataFrame, np.ndarray]): The covariates data.
        response (Union[pd.DataFrame, np.ndarray]): The response data.
        batch (Union[pd.DataFrame, np.ndarray]): The batch effect data.
        train_idx (List[List[int]]): Indices of the training set.
        test_idx (List[List[int]]): Indices of the testing set.
        output_path (str): The path to the output directory.
        log_path (str): The path to the log directory.
        outputsuffix (str): The suffix to add to the output files.
        alg (str): The algorithm to use for fitting the model. Options: 'gpr', 'hbr', 'blr'
        optimizer (str): The optimizer to use for fitting the model.
        save_results (bool): Whether to save the results.
        save_model (bool): Whether to save the model.
        
    Returns:
        pd.DataFrame: The fitted model estimations.
    """
    # if use HBR, you need to provide batch effect data
    if alg == 'hbr' and batch is None:
        raise ValueError('You need to provide batch effect data for Hierarchical Bayesian model.')

    # to store the deviation scores
    all_results = np.zeros((covariates.shape[0], 3))

    # iterate over the folds
    for fold_idx, (slices_train, slices_test) in enumerate(zip(train_idx, test_idx)):
        cov_train = covariates.iloc[slices_train] if isinstance(covariates, pd.DataFrame) else covariates[slices_train]
        cov_test = covariates.iloc[slices_test] if isinstance(covariates, pd.DataFrame) else covariates[slices_test]
        resp_train = response.iloc[slices_train] if isinstance(response, pd.DataFrame) else response[slices_train]
        resp_test = response.iloc[slices_test] if isinstance(response, pd.DataFrame) else response[slices_test]
        with open(f'cov_train_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(cov_train, f)
        with open(f'resp_train_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(resp_train, f)
        with open(f'cov_test_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(cov_test, f)
        with open(f'resp_test_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(resp_test, f)
            
        # TODO: add batch effect data if HBR is used
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
            savemodel=save_model
        )

        # load the deviation scores
        with open(f'Z_{outputsuffix}_fold{fold_idx}.pkl', 'rb') as f:
            Z = pickle.load(f)
        with open(f'yhat_{outputsuffix}_fold{fold_idx}.pkl', 'rb') as f:
            yhat = pickle.load(f)
        with open(f'ys2_{outputsuffix}_fold{fold_idx}.pkl', 'rb') as f:
            s2 = pickle.load(f)
        all_results[slices_test] = np.stack([Z, yhat, s2], axis=1)
    return pd.DataFrame(data=all_results, columns=['Z', 'yhat', 's2'])

def predict_normative_model(covariates: pd.DataFrame, measures: pd.DataFrame, model_path: str, alg: str = 'gpr'):
    """
    Make predictions for unseen subjects using a normative model.
    
    Args:
        covariates (pd.DataFrame): The covariates data.
        measures (pd.DataFrame): The measures data.
        model_path (str): The path to the normative model.
        alg (str): The algorithm to use for prediction.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The predicted mean, the variance, and the deviation scores.
    """
    cov_file = 'test_covariates.pkl'
    resp_file = 'test_measures.pkl'
    with open(cov_file, 'wb') as f:
        pickle.dump(covariates, f)
    with open(resp_file, 'wb') as f:
        pickle.dump(measures, f)
    yhat, s2, Z = predict(cov_file, respfile=resp_file,  alg=alg, outscaler='standardize', model_path=model_path)
    return Z, yhat, s2