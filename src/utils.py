import os
import sys
import dill
import pandas as pd
import numpy as np
from src.logger import CustomException, logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    """
    Evaluate multiple regression models and return their performance metrics.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training target variable
    - X_test: Testing feature set
    - y_test: Testing target variable
    - models: Dictionary of model names and their corresponding instances
    
    Returns:
    - model_report: Dictionary with model names as keys and R2 scores as values
    """
    model_report = {}
    
    for model_name, model in models.items():
        try:
            params[model_name] = params.get(model_name, {})

            gs = GridSearchCV(estimator=model, param_grid=params[model_name], cv=3, n_jobs=-1, verbose=2)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)

            logging.info(f"Training {model_name} with parameters: {gs.best_params_}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_score_value = r2_score(y_test, y_pred)
            model_report[model_name] = r2_score_value
        except Exception as e:
            raise CustomException(f"Error evaluating {model_name}: {e}", sys)
    
    return model_report

def load_object(file_path):
    """
    Load a Python object from a file using dill.
    
    Parameters:
    - file_path: Path to the file containing the serialized object
    
    Returns:
    - The deserialized object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
