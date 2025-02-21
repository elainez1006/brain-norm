import numpy as np
import pandas as pd
import pickle

def predict_normative_model(model_path: str, data: pd.DataFrame, slices_test: List[int]):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model.predict(data.iloc[slices_test])