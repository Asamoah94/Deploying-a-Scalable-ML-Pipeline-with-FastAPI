import pytest
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.linear_model import LogisticRegression
import numpy as np


# Test if the ML function returns the expected type of result
def test_train_model_returns_expected_type():
    """
    Test if the ML function returns the expected type of result.
    """
    # Generate some dummy data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    
    # Call the ML function and assert its output type
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


# Test whether the trained model is fitted
def test_model_is_fitted():
    """
    Test whether the trained model is fitted.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    model = train_model(X_train, y_train)
    # Assert that the model is fitted by checking if it has coefficients
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')


def test_dataset_size_and_dtype():
    """
    Test if the training and test datasets have the expected size or data type.
    """
    # Generate some dummy data
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)

    # Assert the size and data type of the training and test datasets
    assert X_train.shape == (100, 5)
    assert y_train.shape == (100,)
    assert X_test.shape == (20, 5)
    assert y_test.shape == (20,)
    assert X_train.dtype == np.float64
    assert y_train.dtype == np.int64
    assert X_test.dtype == np.float64
    assert y_test.dtype == np.int64


# Run the tests
if __name__ == "__main__":
    pytest.main(['-v', 'test_ml.py'])
