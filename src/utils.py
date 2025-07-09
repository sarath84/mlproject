import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score
from src.exceptions import CustomException
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(file_path,obj):
    dir_path=os.makedirs(os.path.dirname(file_path),exist_ok=True)
    try:
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
            raise CustomException(e,sys)
        
        
        
        
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor # Even if not used in ModelTrainer, its params can be validated
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
# from xgboost import XGBRegressor # type: ignore
# from catboost import CatBoostRegressor         # type: ignore
# from src.logger import logging
        

# def validate_hyperparameters_and_raise_error(models_dict, params_dict):
#     """
#     Validates the hyperparameter keys in the params_dict against the
#     actual valid parameters for each corresponding model.
#     Raises a ValueError if any unrecognized hyperparameter names are found.
#     Collects all errors and presents them in a single, comprehensive message.

#     Args:
#         models_dict (dict): A dictionary where keys are model names (strings)
#                             and values are instantiated model objects.
#         params_dict (dict): A dictionary where keys are model names (strings)
#                             and values are dictionaries of hyperparameters
#                             to be tuned for that model.
#     Raises:
#         ValueError: If any hyperparameter name in params_dict does not
#                     correspond to a valid parameter for its respective model.
#     """
#     all_errors = []

#     for model_name, model_instance in models_dict.items():
#         # Check if this model has a corresponding entry in the params_dict
#         if model_name in params_dict:
#             defined_params_for_model = params_dict[model_name]

#             # If the params dictionary for this model is empty, no validation needed
#             if not defined_params_for_model:
#                 continue

#             try:
#                 # Get the set of all valid parameter names for this model instance
#                 valid_params_for_model = set(model_instance.get_params().keys())
#             except Exception as e:
#                 # This handles cases where calling get_params() might fail (e.g., if a model isn't properly initialized)
#                 all_errors.append(
#                     f"Error checking parameters for model '{model_name}': {e}. "
#                     f"Skipping hyperparameter validation for this model."
#                 )
#                 continue

#             # Check each hyperparameter name defined by the user
#             for param_name_user_defined in defined_params_for_model.keys():
#                 if param_name_user_defined not in valid_params_for_model:
#                     # Construct an informative error message
#                     # Try to suggest valid params if there are similar ones, or just list a few.
                    
#                     # Get a few valid parameters to hint to the user
#                     sample_valid_params_hint = sorted(list(valid_params_for_model))[:min(len(valid_params_for_model), 5)]
#                     hint_str = f"Valid parameters include: {', '.join(sample_valid_params_hint)}..." if sample_valid_params_hint else "No valid parameters found (check model type)."

#                     error_msg = (
#                         f"Invalid Hyperparameter for '{model_name}': "
#                         f"The parameter name '{param_name_user_defined}' is not recognized. "
#                         f"Please check the spelling. {hint_str}"
#                     )
#                     all_errors.append(error_msg)

#     # If the all_errors list is not empty, raise a single ValueError containing all detected issues
#     if all_errors:
#         raise ValueError("\nHyperparameter configuration errors found:\n" + "\n".join(all_errors))
        
        
# def evaluate_models(X_train,Y_train,X_test,Y_test,models,params):
#     try: 
#         report={}
        
#         for i in range (list(models)):
#             model=list(models.value())[i]
#             para=params[models.key()[i]]
            
#             gs=GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,Y_train)
            
#             model.set_params(**gs.best_params_)
#             model.fit(X_train,Y_train)
            
            
#             Y_train_predict=model.predict(Y_train)
            
            
#             Y_test_predict=model.predict(Y_test)
            
#             train_score=r2_score(Y_train,Y_train_predict)
            
#             test_score=r2_score(Y_test,Y_test_predict)
            
#             report[list(models.key())[i]]=test_score
            
            
#             return report    
        
def evaluate_models(X_train, Y_train, X_test, Y_test, models, params):
    try:
        report = {}
        logging.info("Starting model evaluation process...")

        # 1. Correct way to iterate through dictionary items (model_name and model_instance)
        for model_name, model_instance in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # 2. Correct way to get parameters for the current model
            # Use .get() to avoid KeyError if a model name in 'models' doesn't have params in 'params'
            param_grid = params.get(model_name)

            # Initialize GridSearchCV only if parameters are provided for the current model
            if param_grid:
                logging.info(f"  Applying GridSearchCV with parameters: {param_grid}")
                # Use n_jobs=-1 for parallel processing, verbose=0 to suppress GridSearch internal logging
                gs = GridSearchCV(model_instance, param_grid, cv=3, verbose=0, n_jobs=-1)
                gs.fit(X_train, Y_train) # Fit GridSearchCV
                
                # Get the best estimator found by GridSearchCV
                best_model = gs.best_estimator_
                logging.info(f"  GridSearchCV best parameters for {model_name}: {gs.best_params_}")
            else:
                logging.info(f"  No specific parameters for {model_name}. Training without GridSearchCV.")
                best_model = model_instance # Use the model instance directly if no params

            # Train the chosen model (either from GridSearchCV or original instance)
            best_model.fit(X_train, Y_train)

            # 3. Corrected: Make predictions on X_train and X_test (features), not Y_train/Y_test (targets)
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(Y_train, y_train_pred)
            test_model_score = r2_score(Y_test, y_test_pred)

            logging.info(f"  {model_name} - Train R2 Score: {train_model_score:.4f}, Test R2 Score: {test_model_score:.4f}")

            # Store the test score in the report using the model's name
            report[model_name] = test_model_score
            
        # 4. CRITICAL: 'return report' must be outside the loop
        # It must only return AFTER all models have been evaluated.
        logging.info("Model evaluation completed for all models.")
        return report
    except Exception as e:
        raise CustomException(e,sys)      
        