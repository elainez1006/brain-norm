# test_data_utils.py
import pytest
import pandas as pd
from brainnorm.utils import read_dataframe, split_dataframe


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'subject_id': ['sub-01', 'sub-01', 'sub-02', 'sub-03'],
        'value1': [1.0, 3.0, 5.0, 7.0],
        'value2': [2.0, 4.0, 6.0, 8.0]
    })

def test_read_dataframe(tmp_path):
    # Create a test CSV
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    # Test reading
    result = read_dataframe(str(csv_path))
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)

def test_split_dataframe(sample_df):
    slices_train, slices_test = split_dataframe(
        data=sample_df,
        folds=3,
        id_col='subject_id'
    )
    
    assert len(slices_train) == 3
    assert len(slices_test) == 3
    
    # Check train and test are different
    for train, test in zip(slices_train, slices_test):
        assert set(train).isdisjoint(set(test))
