import sys
from dataclasses import dataclass
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This method creates a data transformation pipeline that includes:
        """ 
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("Data transformation pipelines created successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error("Error occurred while creating data transformation pipelines.")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method starts")
        try:
            logging.info("Creating DataFrames.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("DataFrames created successfully.")

            logging.info("Applying preprocessing object.")
            preprocessor = self.get_data_transformer_object()

            target_column = 'math_score'
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise CustomException(f"Target column '{target_column}' not found in the datasets.", sys)
            
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logging.info("Preprocessing completed.")

            logging.info("Combining train and test arrays.")
            train_array = np.c_[X_train, np.array(y_train)]
            test_array = np.c_[X_test, np.array(y_test)]
            logging.info("Combining completed.")

            logging.info("Saving preprocessed objects.")
            save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessor
                )
            logging.info("Saving preprocessed objects completed.")

            return (train_array, 
                    test_array,
                    self.data_transformation_config.preprocessor_obj_file_path,
                    )

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys)