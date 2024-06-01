import pytest
from ml.model import train_model, compute_model_metrics
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


# Test if the ML model uses the expected algorithm
def test_train_model_uses_expected_algorithm():
    """
    Test if the ML model uses the expected algorithm.
    """
    # Generate some dummy data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    
    # Call the ML function and assert its algorithm
    model = train_model(X_train, y_train)
    assert model.__class__.__name__ == 'LogisticRegression'


# Test the train_model function with a larger dataset
def test_train_model_with_more_data():
    """
    Test the train_model function with a larger dataset.
    """
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)






# Run the tests
if __name__ == "__main__":
    pytest.main(['-v', 'test_ml.py'])

