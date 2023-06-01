import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn .model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path= os.path.join("artifacts","test.csv")
    raw_data_path= os.path.join("artifacts","raw.csv")
    
  #notebook\data\adult.csv  
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        
        logging.info("Data ingestion is started!!")
        try:
            logging.info("Data Reading using Pandas library local system")
            data=pd.read_csv(os.path.join("notebook\data","adult.csv"))
            
            logging.info("Data reading completed!!")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index= False)
            
          
            logging.info("DData is splitted in train and test!!")
            train_set,test_set = train_test_split(data,test_size=0.30,random_state=42)
            
                       
            #save data in artifact folder
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header = True)
            
            logging.info("Data ingestion completed!!")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path 
            )
            
        except Exception as e:
            logging.info("Error occurred in Data Ingestion Stage")
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    
    #src\components\data.ingestion.py