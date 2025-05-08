import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Custom cleaning steps before pipeline processing.
        """
        try:
            # Replace special characters with NaN and fix FutureWarning
            special_chars = {'#', '@', '+', '$', '*'}
            df = df.replace(special_chars, np.nan)
            df = df.infer_objects(copy=False)

            # Convert numeric columns safely
            numeric_columns_to_convert = [
                'Tenure', 'Account_user_count', 'rev_per_month',
                'rev_growth_yoy', 'coupon_used_for_payment', 'Day_Since_CC_connect'
            ]
            df[numeric_columns_to_convert] = df[numeric_columns_to_convert].apply(pd.to_numeric, errors='coerce')

            # Impute numeric columns using KNN
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_columns_to_convert] = imputer.fit_transform(df[numeric_columns_to_convert])

            # Round and convert to int where appropriate
            integer_columns = ['Tenure', 'Account_user_count', 'Day_Since_CC_connect', 'coupon_used_for_payment']
            df[integer_columns] = df[integer_columns].round(0).astype(int)

            # Clean categorical values
            df['account_segment'] = df['account_segment'].replace({
                'Regular +': 'Regular Plus',
                'Super +': 'Super Plus'
            })

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numercial_columns = ['Tenure', 'City_Tier', 'CC_Contacted_LY', 'Service_Score', 'Account_user_count', 'CC_Agent_Score', 'rev_per_month', 'Day_Since_CC_connect', 'cashback']
            categorical_columns = ['Payment', 'Gender', 'account_segment', 'Login_device']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer()),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numercial_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numercial_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_excel(train_path)
            test_df = pd.read_excel(test_path)

            logging.info("Read train and test data completed")

            logging.info("Cleaning and preprocessing data")
            train_df = self.clean_and_preprocess(train_df)
            test_df = self.clean_and_preprocess(test_df)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"
            numercial_columns = ['Tenure', 'City_Tier', 'CC_Contacted_LY', 'Service_Score', 'Account_user_count', 'CC_Agent_Score', 'rev_per_month', 'Day_Since_CC_connect', 'cashback']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
