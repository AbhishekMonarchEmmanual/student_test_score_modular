import pandas as pd 
import numpy as np 
from source.exception import SystemException
from source.logger import logging
import os, sys
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from source.config.config import *
from source.config.atrifact import *
from source.config import *
from source.utilities.utilities import *

class DataTransformation:
  
    def __init__(self, data_transformation_config:DataTransformationConfig, artifact_file=DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            
        except Exception as e:
            raise SystemException(e,sys)
              
            
    def initiate_data_transformation(self, train_path:DataTransformationConfig,test_path:DataTransformationConfig):
        try:
            
            logging.info(f"________DATA TRANSFORMATIO BEGINS______________")
            logging.info("we are creating train and test data frame before preprocessing")
            train_df = pd.read_csv(self.data_transformation_config.train_data_path)
            test_df = pd.read_csv(self.data_transformation_config.test_data_path)
            logging.info("we are completed creating dataframes")
            
            preprocessing_obj = get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["reading_score","writing_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"Applying the preprocessing object on training dataframe and testing platform"
            )
            
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.fit_transform(input_feature_test_df)
            
            logging.info(f"shape of  input_feature_train_arr : {input_feature_train_arr.shape}")
            logging.info(f"shape of input_feature_test_arr : {input_feature_test_arr.shape}")
            
            logging.info(f"shape of test_arr {np.array(target_feature_train_df).shape}")
            logging.info(f"shape of test_arr {np.array(target_feature_test_df).shape}")
            
            logging.info(f"_______Combining array__________")
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"shape of train_arr {train_arr.shape}")
            logging.info(f"shape of test_arr {test_arr.shape}")
            
            
           
            
            logging.info(f"_________PREPARING FOR SAVING OBJECT AND PICKLE FILE_____________")
            
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"_________SAVING NUMPY ARRAY_____________")
            np.save(self.data_transformation_config.transformed_train_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_path, test_arr)
            
            logging.info(f"_________PREPARING DATA TRANSFORMATION ARTIFACT_____________")
            
            data_transformation_artifact= DataTransformationArtifact(
                object_path=self.data_transformation_config.preprocessor_obj_file_path, 
                train_arr=self.data_transformation_config.transformed_train_path, 
                test_arr=self.data_transformation_config.transformed_test_path
                )
            
            logging.info(f"_______DATA TRANSFORMATION COMPLETE______")
            logging.info(f"object path : {data_transformation_artifact.object_path}, train arr path : {data_transformation_artifact.train_arr}, test arr path {data_transformation_artifact.test_arr}")
            
            
            return  data_transformation_artifact
            
        except Exception as e:
            raise SystemException(e,sys)       
        
  
        
        