from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



application = Flask(__name__)

app = application

# Route to render index.html template
@app.route("/")

def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('Ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score')),
        )
        
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        predict = predict_pipeline.predict(pred_df)
        return render_template('home.html', prediction = predict[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')