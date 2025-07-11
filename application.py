from flask import Flask,request,render_template
import sys
from src.exceptions import CustomException
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from src.pipeline.predict_pipline import Predict_Pipeline,CustomData

application= Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])    
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(gender=request.form.get('gender'),
                        race_ethnicity=request.form.get('ethnicity'),
                        parental_level_of_education=request.form.get('parental_level_of_education'),
                        lunch=request.form.get('lunch'),
                        test_preparation_course=request.form.get('test_preparation_course'),
                        reading_score=float(request.form.get('reading_score')),
                        writing_score=float(request.form.get('writing_score')))
        
        predict_data_frame=data.get_data_as_data_frame()
        print(predict_data_frame)
        
        print("data before prediction")   
        
        prediction_pipeline=Predict_Pipeline()
        
        print("mid phase of prediction")
        
        result=prediction_pipeline.predict(predict_data_frame)
        
        print("after prediction")
        formated_result=f"{result[0]:.2f}"
        return render_template('home.html',result=formated_result)
    
application.debug = True         
if __name__=="__main__":
    application.run(host="0.0.0.0")              