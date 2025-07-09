import sys
import os
try:
    import catboost
    catboost_version = catboost.__version__
    catboost_path = catboost.__file__
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(verbose=False)
    catboost_params = model.get_params().keys()
except ImportError:
    catboost_version = "Not Installed"
    catboost_path = "N/A"
    catboost_params = "N/A (CatBoost not imported)"
except Exception as e:
    catboost_version = "Error during import/init"
    catboost_path = f"Error: {e}"
    catboost_params = f"Error: {e}"

print("--- Environment Diagnosis ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Conda prefix: {os.environ.get('CONDA_PREFIX', 'Not set (Not in Conda env)')}")
print(f"Active Prompt Name: {os.environ.get('CONDA_DEFAULT_ENV', 'No Conda env active')}")
print("\n--- CatBoost Diagnosis ---")
print(f"CatBoost Version: {catboost_version}")
print(f"CatBoost Path (__file__): {catboost_path}")
print(f"CatBoostRegressor Parameters: {list(catboost_params)}") # Convert to list for clear printing
print("---------------------------")