from flask import Flask, request, render_template
import sys
import os # Import os for path checks
import numpy as np
import pandas as pd
import logging # Import logging
# import pickle # Uncomment if you're directly loading .pkl files in this file

# Initialize logging BEFORE anything else that might fail
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming src is in your project root, make sure it's discoverable
# Elastic Beanstalk typically handles Python path for standard project structures,
# but if you face ModuleNotFoundError for src, you might need to ensure your
# deployment bundle includes src correctly or adjust paths.
# Example if src is a sibling directory to application.py:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.exceptions import CustomException
    # Assuming Predict_Pipeline and CustomData are defined within predict_pipline.py
    from src.pipeline.predict_pipline import Predict_Pipeline, CustomData
    logging.info("Successfully imported modules from src.")
except ImportError as e:
    logging.error(f"Failed to import modules from src: {e}", exc_info=True)
    sys.exit(1) # Critical error: cannot proceed without core modules
except Exception as e:
    logging.error(f"An unexpected error occurred during src imports: {e}", exc_info=True)
    sys.exit(1)

application = Flask(__name__)
app = application # Assigning application to app for consistency with route decorators

# Set Flask's logger level
application.logger.setLevel(logging.DEBUG)

# --- Global Scope Model/Preprocessor Loading (Crucial for identifying early errors) ---
# It's best practice to load models ONCE when the app starts, not on every request.
# If your Predict_Pipeline handles loading internally, you might not need these
# global loading lines here, but ensure Predict_Pipeline's __init__ or predict method
# handles file loading robustly with error handling and logging.

# Example if you need to load models directly here:
# global_model = None
# global_preprocessor = None

# try:
#     application.logger.info("Attempting to load model and preprocessor (if applicable)...")
#     # UNCOMMENT AND ADJUST THESE LINES IF YOU HAVE model.pkl and preprocessor.pkl
#     # Make sure the paths are correct relative to where application.py is deployed.
#     # For example, if they are in a 'artifacts' folder next to application.py:
#     # model_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'model.pkl')
#     # preprocessor_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'preprocessor.pkl')

#     # if not os.path.exists(model_path):
#     #     application.logger.error(f"Model file not found: {model_path}")
#     #     # Consider raising an error if the model is critical
#     # if not os.path.exists(preprocessor_path):
#     #     application.logger.error(f"Preprocessor file not found: {preprocessor_path}")
#     #     # Consider raising an error if the preprocessor is critical

#     # with open(model_path, 'rb') as f:
#     #     global_model = pickle.load(f)
#     # with open(preprocessor_path, 'rb') as f:
#     #     global_preprocessor = pickle.load(f)

#     application.logger.info("Model and preprocessor loaded successfully (if applicable).")

# except FileNotFoundError as e:
#     application.logger.error(f"FileNotFoundError during global model/preprocessor loading: {e}", exc_info=True)
#     sys.exit(1) # Exit if critical resources are missing at startup
# except Exception as e:
#     application.logger.error(f"General error during global model/preprocessor loading: {e}", exc_info=True)
#     sys.exit(1) # Exit if critical resources cannot be loaded


# --- Routes with enhanced logging and error handling ---

@app.route('/')
def index():
    application.logger.info("Accessing root route: /")
    try:
        # Check if 'templates' directory and 'index.html' exist before rendering
        # This helps catch FileNotFoundError earlier
        template_path = os.path.join(application.root_path, 'templates', 'index.html')
        if not os.path.exists(template_path):
            application.logger.error(f"Template file not found: {template_path}")
            return "Internal Server Error: index.html not found.", 500

        return render_template('index.html')
    except Exception as e:
        application.logger.error(f"Error rendering index.html: {e}", exc_info=True)
        # Return a meaningful error to the user/health checker
        return "Internal Server Error: Could not load index page. Check logs for details.", 500

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    application.logger.info(f"Accessing predictdata route via {request.method} method.")
    if request.method == 'GET':
        try:
            template_path = os.path.join(application.root_path, 'templates', 'home.html')
            if not os.path.exists(template_path):
                application.logger.error(f"Template file not found: {template_path}")
                return "Internal Server Error: home.html not found.", 500
            return render_template('home.html')
        except Exception as e:
            application.logger.error(f"Error rendering home.html for GET: {e}", exc_info=True)
            return "Internal Server Error: Could not load prediction form. Check logs for details.", 500
    else: # POST request
        try:
            application.logger.info("Received POST request data for prediction.")
            # Data parsing and validation
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            
            # Type conversion with validation
            try:
                reading_score = float(request.form.get('reading_score'))
                writing_score = float(request.form.get('writing_score'))
            except (ValueError, TypeError) as e:
                application.logger.error(f"Invalid score format: {e}", exc_info=True)
                return "Input Error: Reading and writing scores must be valid numbers.", 400

            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            predict_data_frame = data.get_data_as_data_frame()
            application.logger.debug(f"Input DataFrame for prediction: \n{predict_data_frame.to_string()}") # .to_string() for better logging

            application.logger.info("Initializing prediction pipeline...")
            # Ensure Predict_Pipeline's __init__ doesn't have hidden errors
            prediction_pipeline = Predict_Pipeline() 

            application.logger.info("Starting prediction process...")
            result = prediction_pipeline.predict(predict_data_frame)
            application.logger.info("Prediction completed successfully.")

            formated_result = f"{result[0]:.2f}"
            application.logger.info(f"Prediction result: {formated_result}")
            return render_template('home.html', result=formated_result)

        except CustomException as e: # Catch your specific custom exceptions
            application.logger.error(f"CustomException during prediction: {e}", exc_info=True)
            return f"Prediction Error: {e}", 500
        except Exception as e:
            application.logger.error(f"Unhandled exception during prediction POST request: {e}", exc_info=True)
            return "Internal Server Error: An unexpected error occurred during prediction. Check logs.", 500

# The __name__ == "__main__" block should generally be commented out or removed
# for Elastic Beanstalk deployments as Gunicorn handles running the application.
# Keeping it can sometimes lead to unexpected behavior if not handled carefully.
if __name__ == "__main__":
    # For local development, you can uncomment this line
    # debug=True should NOT be used in production environments
    application.run(host="0.0.0.0", debug=True)