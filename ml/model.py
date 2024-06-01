import sys
import os
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    
    Returns
    -------
    model : LogisticRegression
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model : LogisticRegression or OneHotEncoder
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Loads pickle file from `path` and returns it.
    
    Inputs
    ------
    path : str
        Path to load pickle file from.
    
    Returns
    -------
    model : LogisticRegression or OneHotEncoder
        Loaded machine learning model or OneHotEncoder.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics on a slice of the data specified by a column name and value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : LogisticRegression
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    slice_data = data[data[column_name] == slice_value]
    X_slice, y_slice, _, _ = process_data(
        slice_data, categorical_features, label, training=False, encoder=encoder, lb=lb)
        
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

