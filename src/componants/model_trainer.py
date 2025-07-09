import os
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from sklearn.ensemble import(AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from dataclasses import dataclass




@dataclass

class ModelTrainerConfigg:
    model_trainer_config=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_file_path=ModelTrainerConfigg()
        
        
    def Initiate_Model_Trainer(self,train_arr,test_arr)   :
        
        try:
           
            
            
            
            logging.info(" splitting all the  training and test array as features to train and test and target to train and test where X--> input feature , Y-->input target")
            X_train,Y_train,X_test,Y_test=(
               train_arr[:,:-1],
               train_arr[:,-1],
               test_arr[:,:-1],
               test_arr[:,-1]
            ) 
            logging.info("applying ml algorith is started")
            models={
               "random_forest":RandomForestRegressor(),
               "Descision_tree":DecisionTreeRegressor(),
               "gradient_boosting":GradientBoostingRegressor(),
               "linear_regression":GradientBoostingRegressor(),
               "XGBresgressor":XGBRegressor(),
               "CatBoosting Regressor": CatBoostRegressor(verbose=False),
               "AdaBoost Regressor": AdaBoostRegressor(),
               
               
               
           }
           
            parameters={
               "Descision_tree":{
                   'criterion':['squared_error','friedman_mse','absolute_error','poisson']
               },
               "Random Forest":{'n_estimators':[8,16,32,64,128,256]},
               
               "Gradient Boosting":{'learning_rate':[.1,.01,.05,.001],
                                    'susample'    :[0.6,0.7,0.75,0.8,0.85,0.9],
                                    'n_estimator':[8,16,32,64,128,256]},
               "Linear Regresion":{},
               "XGBregresor":{'learning_rate':[.1,.01,.05,.001],
                               'n_estimator':[8,16,32,64,128,256],
                               },
               "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
               },
               "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
               
           }
            logging.info("validating hyperparameters")
            # validate_hyperparameters_and_raise_error(models, parameters) # Pass the local 'models' and 'params'
            # logging.info("Hyperparameters validated successfully. Proceeding with evaluation.")   
           
           
            model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,
                                             models=models,params=parameters)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_file_path.model_trainer_config,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_sc = r2_score(Y_test, predicted)
            return r2_sc
            
            
        except Exception as e:
            raise CustomException(e,sys)
            
    
