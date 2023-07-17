import pandas as pd 
import numpy as np 
import streamlit as st
import dill
from source.exception import SystemException
from source.logger import logging
import os, sys 
import pandas as pd 
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from source.components.data_ingestion import *
from source.config.config import *
from source.components.data_transformation import *
from source.components.model_trainer import *
from source.config.atrifact import *
from source.utilities.utilities import *
from source.list_model.list_class import *



class initating_complete_training:
    
    def __init__(self):
        pass
    def initiate_training(self):
        try :
            training_pipeline = Training_Pipeline()
            data_ingestion_config = DataIngestionConfig(training_pipeline)


            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
            print(data_ingestion_artifact)

            data_transformation_config= DataTransformationConfig(training_pipeline=training_pipeline,data_artifacts_file_paths=data_ingestion_artifact )
            data_transformation= DataTransformation(data_transformation_config=data_transformation_config, artifact_file=data_ingestion_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation(train_path=data_ingestion_artifact.train_data_path,test_path=data_ingestion_artifact.test_data_path)
            print(data_transformation_artifact)


            model_trainer_config = ModelTrainerConfig(training_pipeline=training_pipeline, datatransformationartifact=data_transformation_artifact)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config)
            _,modeltrainerartifact = model_trainer.initiate_model_trainer(train_arr=model_trainer_config.train_arr, test_arr=model_trainer_config.test_arr) 
            print(modeltrainerartifact.object_path)
            list_dirs = new_path_collector(model_path= modeltrainerartifact, scaled_obj=data_transformation_artifact)
            model_path , transform_path = list_dirs.list_creation()
            model_pred = load_object(model_path[0])
            preprocessor_obj = load_object(transform_path[0])
            
            return model_path, transform_path
        
        except Exception as e:
            raise SystemException(e,sys)


            
            
            
            
        
            
            
            
            
            
        
        
        
        
        
        
