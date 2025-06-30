import sys
import os
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DatainjectionConfigg:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path : str =os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')
    
class DataInjection:
    def __init__(self):
        self.Injection_Configg=DatainjectionConfigg()
    
    def Initiate_Data_injection(self):
        logging.info("data injection started")    
        
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("dataset read")
            os.makedirs(os.path.dirname(self.Injection_Configg.train_data_path),exist_ok=True)
            df.to_csv(self.Injection_Configg.raw_data_path,index=False,header=True)
            logging.info("data splitting has started")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.Injection_Configg.train_data_path,index=False,header=True)
            test_set.to_csv(self.Injection_Configg.test_data_path,index=False,header=True)
            return(
                self.Injection_Configg.test_data_path,
                self.Injection_Configg.train_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__' :
    obj=DataInjection()
    obj.Initiate_Data_injection()       
        
            
        