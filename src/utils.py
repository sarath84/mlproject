import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exceptions import CustomException

def save_object(file_path,obj):
    dir_path=os.makedirs(os.path.dirname(file_path),exist_ok=True)
    try:
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
            raise CustomException(e,sys)