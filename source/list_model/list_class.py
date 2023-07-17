from source.exception import *
from source.logger import * 
import os, sys 
import pandas as pd 
from datetime import datetime
from source.config.atrifact import *
from sklearn.preprocessing import OneHotEncoder,StandardScaler



class new_path_collector:
    def __init__(self,model_path:ModelTrainerArtifact, scaled_obj:DataTransformationArtifact ):
        
        self.model_path = model_path.object_path
        self.scaled_obj = scaled_obj.object_path
        
    def list_creation(self):
        model_list = []
        scaled_obj_list = []
        
        model_list.append(self.model_path)
        scaled_obj_list.append(self.scaled_obj)
        
        return model_list, scaled_obj_list




