from templates import input_webpage
import streamlit as st
import pandas as pd 
import numpy as np 
import dill 
import os ,sys 
from source.exception import SystemException
from source.logger import logging
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



class Prediction_pipeline:
    def __init__(self, model_path, transform_obj, df:pd.DataFrame):
       
        
        try:
            self.model_path = model_path
            self.transform_obj = transform_obj
            self.df = df
            self.training_pipeline = Training_Pipeline()
        except Exception as e:
            raise SystemException(e,sys)
        
        
    def initate_prediction(self):
        try:
            model1 = load_object(self.model_path)
            transform_obj = load_object(self.transform_obj)
            scaled_data = transform_obj.transform(self.df)
            preds = model1.predict(scaled_data)
            return preds     
        except Exception as e:
            raise SystemException(e,sys)   


    
    
    
