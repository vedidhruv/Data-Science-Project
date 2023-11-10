import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_ob_filepath = os.path.join('artefacts', 'preprocessor_ob.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_onject(self):
        '''
        This function is responsible for creating a data transformer object
        '''
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = [
                'gender',
                'lunch',
                'parental level of education',
                'test preparation course',
                'race/ethnicity'
                ]
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('std_scaler', StandardScaler())
                ]
            )
            logging.info("Categorical and numerical pipelines created")
            preprocessor = ColumnTransformer(
                [
                    ("nump_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
            
        except Exception as e:
            logging.error("Data transformation failed")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data read as DataFrame")
            logging.info("Obtaining Preprocessing object")
            
            preprocessing_object = self.get_data_transformer_onject()
            target_column_name = 'math score'
            numerical_features = ['reading score', 'writing score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on train dataframe and test dataframe")
            
            input_feature_train_arr = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_arr = test_df[target_column_name]
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved Preprocessing object at {self.data_transformation_config.preprocessor_ob_filepath}")
            
            save_obj(
                file_path = self.data_transformation_config.preprocessor_ob_filepath,
                obj = preprocessing_object
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_filepath
            )
        except Exception as e:
            raise CustomException(e, sys)