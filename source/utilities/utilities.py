import pandas as pd 
import numpy as np 
from source.exception import SystemException
from source.logger import logging
import os, sys
import dill
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from source.config.config import *
from source.config.atrifact import *
from source.config import *
from dataclasses import dataclass
from sklearn.metrics import r2_score


def get_data_transformer_object():
    """_summary_
    use for transofrming the data frame for prediction

    Returns:
        scaling obje: it will scale and transform our data based on the pipeline we have created 
    """
    try:
    
        numerical_columns =["reading_score","writing_score"] 
        categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
        
        num_pipeine = Pipeline(
            
            steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
            ]
        )
        
        cat_pipeline= Pipeline(
      
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ]
        )
        
        preprocessor = ColumnTransformer(
            [
                ("num_pipeine", num_pipeine,numerical_columns),
                ("cat_piplines",cat_pipeline,categorical_columns)
                
            ]
        )
 
        return preprocessor
 
    except Exception as e:
        raise SystemException(e,sys)
    

def save_object(file_path:str, obj):
    """Use for saving object and pickle file require dill 

    Args:
        file_path (str): path you like to save your model
        obj (_type_): pkl object name

    Raises:
        SystemException: _description_
    """
    
    try : 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise SystemException(e,sys)
    
    
def load_object(file_path: str, ) -> object:
    """use to load pkl file

   
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SystemException(e, sys) from e
    
   
def evaluate_model(X_train,Y_train,X_test,Y_test, models):
    """_summary_
    this function we will use to see which model works best for us

    Args:
        X_train (_type_): train dataset
        Y_train (_type_): train data set
        X_test (_type_): test data set
        Y_test (_type_): test data set
        models (_type_): list of dictionary of {models} ex : 
        {"linear_regression": Linear_Regression(), 'xgboost':XGBoost()...and so on}
        
        

    Raises:
        SystemException: _description_

    Returns:
        _best report in form of dictionary of model name with r2_scores
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,Y_train)
            
            Y_train_pred = model.predict(X_train)
            Y_test_pred =model.predict(X_test)
            
            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(Y_test,Y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise SystemException(e,sys)
    
            
            