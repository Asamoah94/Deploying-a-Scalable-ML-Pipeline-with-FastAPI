import pytest
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
from sklearn.tree import DecisionTreeClassifier
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
    assert isinstance(model, DecisionTreeClassifier)


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
    assert model.__class__.__name__ == 'DecisionTreeClassifier'


# Test if the computing metrics functions return the expected value
def test_compute_model_metrics_returns_expected_value():
    """
    Test if the computing metrics functions return the expected value.
    """
    # Generate some dummy data
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    
    # Call the compute_model_metrics function and assert its output
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert pytest.approx(precision) == 0.75
    assert pytest.approx(recall) == 0.6666666666666666
    assert pytest.approx(f1) == 0.7058823529411765


# Run the tests
if __name__ == "__main__":
    pytest.main(['-v', 'test_ml.py'])

