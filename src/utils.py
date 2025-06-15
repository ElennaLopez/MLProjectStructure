import os
import sys
import dill
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
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
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_score_value = r2_score(y_test, y_pred)
            model_report[model_name] = r2_score_value
        except Exception as e:
            raise CustomException(f"Error evaluating {model_name}: {e}", sys)
    
    return model_report

