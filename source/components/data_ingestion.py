import pandas as pd 
import numpy as np 
from source.exception import SystemException
from source.logger import logging
import os, sys
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from source.config.config import *
from source.config.atrifact import *



    
class DataIngestion: 
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.ingestion_config = data_ingestion_config
        
    def initiate_data_ingestion(self):
        logging.info("_____INITIATING DATA INGESTION________")
        try : 
            df = pd.read_csv('stud.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False , header = True)
            logging.info(f"RAW FILE CREATED")

            logging.info("train and test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False , header = True)
            
            logging.info("ingestion of the data is completed")

            data_ingestion_artifact = DataIngestionArtifact(
                raw_data_path=self.ingestion_config.raw_data_path,
                train_data_path=self.ingestion_config.train_data_path, 
                test_data_path=self.ingestion_config.test_data_path)
            logging.info("_____INGESTION PROCESS COMPLETED_______")
            return data_ingestion_artifact
            
            
        except Exception as e:
            raise SystemException(e,sys)
        
