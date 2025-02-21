import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from typing import Tuple, List


def read_dataframe(data_path: str) -> pd.DataFrame:
    """
    Read input data from a CSV file and remove any rows with missing values.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The data read from the CSV file.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"The file {data_path} does not exist.")
    data = pd.read_csv(data_path)
    data.dropna(inplace=True)
    return data

def split_dataframe(data: pd.DataFrame, folds: int, id_col: str, random_state: int = 42) -> Tuple[List[int], List[int]]:
    """
    Split the data into training and testing sets.
    
    Args:
        data (pd.DataFrame): The dataframe to split.
        folds (int): The number of folds to split the data into.
        id_col (str): The column containing the subject IDs to use for splitting the data.
        random_state (int): The random state to use for splitting the data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and testing sets.
    """
    subject_index, _ = pd.factorize(data[id_col])
    unique_subject_index = np.unique(subject_index)
    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    
    slices_train = []
    slices_test = []
    
    for _, (train_idx, test_idx) in enumerate(kf.split(unique_subject_index)):
        unique_ids_train = unique_subject_index[train_idx]
        unique_ids_test = unique_subject_index[test_idx]
        slices_train.append([k for k, subject in enumerate(subject_index) if subject in unique_ids_train])
        slices_test.append([k for k, subject in enumerate(subject_index) if subject in unique_ids_test])

    return slices_train, slices_test