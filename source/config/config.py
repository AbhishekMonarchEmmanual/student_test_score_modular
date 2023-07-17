from source.exception import *
from source.logger import * 
import os, sys 
import pandas as pd 
from datetime import datetime
from source.config.atrifact import *
from sklearn.preprocessing import OneHotEncoder,StandardScaler

class Training_Pipeline:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise SystemException(e,sys) 
    
    

class DataIngestionConfig:
    def __init__(self,training_pipeline:Training_Pipeline):
        
        try:
            self.train_data_path: str = os.path.join(training_pipeline.artifact_dir, "complete_data","train.csv")
            self.test_data_path: str = os.path.join(training_pipeline.artifact_dir,"complete_data", 'test.csv')
            self.raw_data_path: str = os.path.join(training_pipeline.artifact_dir,"complete_data", "data.csv")
        except Exception as e :
            raise SystemException(e,sys)

class DataTransformationConfig:
    def __init__(self, training_pipeline: Training_Pipeline, data_artifacts_file_paths:DataIngestionArtifact):
        try:   
            self.preprocessor_obj_file_path = os.path.join(training_pipeline.artifact_dir, "transformed_data","preprocessor.pkl")
            self.train_data_path = data_artifacts_file_paths.train_data_path
            self.test_data_path = data_artifacts_file_paths.test_data_path
            self.transformed_train_path: str = os.path.join(training_pipeline.artifact_dir, "transformed_data","train.npy")
            self.transformed_test_path: str = os.path.join(training_pipeline.artifact_dir,"transformed_data", 'test.npy')
            self.transformed_raw_data_path: str = os.path.join(training_pipeline.artifact_dir,"transformed_data", "data.npy")
            
        except Exception as e:
            raise SystemException(e,sys)

class ModelTrainerConfig:
    def __init__(self, training_pipeline :Training_Pipeline, datatransformationartifact:DataTransformationArtifact):
        self.data_transformation_config = datatransformationartifact
        self.training_pipeline = training_pipeline
        self.train_arr = datatransformationartifact.train_arr
        self.test_arr = datatransformationartifact.test_arr
        self.model_save_path = os.path.join(training_pipeline.artifact_dir, "model_trainer", "model.pkl")
        

                 