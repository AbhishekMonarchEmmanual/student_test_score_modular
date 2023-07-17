from source.exception import *
from source.logger import * 
import os, sys 
import pandas as pd 
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str 
    train_data_path: str
    test_data_path: str 
   
   
@dataclass
class DataTransformationArtifact:
    object_path:str
    train_arr:str
    test_arr:str
    
@dataclass
class ModelTrainerArtifact:
    object_path:str