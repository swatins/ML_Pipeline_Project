# handle missing val,outliers,imbalanced data,cat columns

import os,sys
import pandas as pd
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

 
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info("Data transformation started")
            
            numerical_features=['age', 'workclass', 'education.num', 'marital.status', 'occupation',
         'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week']
          # pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                      
                    ]                                  
                                    )
        # pipeline for numerical features
           # cat_pipeline = Pipeline(
           #     steps=[
            #        ("imputer",SimpleImputer(strategy='mode')),
             #          ]                                  
              #                      )
            
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_features)
            ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys) 
        
    def remove_outliers_IQR(self, col, df):
        try: 
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
           # print("q1=",Q1)
           # print("q1=",Q3)
             #logging.info("")           
            #Q1 = np.quantile(df[col], 0.25 )
            #Q3 = np.quantile(df[col], 0.75 )
            iqr = int(Q3) - int(Q1)

            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            
            df.loc[(df[col]>upper_limit),col]=upper_limit
            df.loc[(df[col]<lower_limit),col]=lower_limit
            
            return df
 
        except Exception as e:
            logging.info("Outliers handling code")
            raise CustomException (e,sys)   
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_Data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            
            numerical_features= ['age', 'workclass', 'education.num', 'marital.status', 'occupation',
                                    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                                    'hours_per_week']
            
            for col in numerical_features:
                self.remove_outliers_IQR(col=col,df=train_Data)
                
            logging.info("Outliers capped on our train data")
            
            for col in numerical_features:
                self.remove_outliers_IQR(col=col,df= test_data)
                
            logging.info("Outliers capped on our test data")
            
            preprocessor_obj= self.get_data_transformation_obj()
            
            target_column= "income"
            drop_column=[target_column]
            
            logging.info("Splitting train data into dependant and independant features")
            input_feature_train_data=train_Data.drop(drop_column,axis=1)
            target_feature_train_data=train_Data[target_column]
            
            logging.info("Splitting test data into dependant and independant features")
            input_feature_test_data=test_data.drop(drop_column,axis=1)
            target_feature_test_data=test_data[target_column]
            
            #apply transformation on train and test data
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)
            
            # preprocessor obj on our train and test data
            
            train_array= np.c_[input_train_arr,np.array(target_feature_train_data)]
            test_array=np.c_[input_test_arr,np.array(target_feature_test_data)]
       
       
            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path,
                        obj=preprocessor_obj )
       
            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocess_obj_file_path)
            
            
            
        except Exception as e:
             raise CustomException(e,sys)      