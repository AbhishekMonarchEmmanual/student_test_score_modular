import streamlit as st
import pandas as pd 
import os
from templates.input_webpage import *
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
from source.pipeline.prediction_pipeline import *
from source.pipeline.training_pipeline import *
from templates.input_webpage import *


                
if __name__ == "__main__":
    webpage  = Prediction_input()                
    title_input_page = webpage.my_title_page()
    if st.checkbox("creating model and predciting values"):
            train_pipe = initating_complete_training()
            model_obj, transform_obj  = train_pipe.initiate_training()
            df = pd.read_csv('templates/predict_df/df.csv')

            pred_pipe = Prediction_pipeline(model_path=model_obj[0], transform_obj=transform_obj[0], df = df)
            
            r2_scores = pred_pipe.initate_prediction()
            df["Predicted_math_scores"] = r2_scores
            if st.checkbox("see your predicted values"):
                st.write(df)
            

                st.text(r2_scores)
        

        
    
    

