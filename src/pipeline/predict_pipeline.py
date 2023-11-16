import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj



class PredictPipline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "artefacts/model.pkl"
            preprocess_path = "artefacts/preprocessor_ob.pkl"
            model = load_obj(file_path = model_path)
            preprocessor = load_obj(file_path = preprocess_path)
            print(preprocessor)
            data_scaled = preprocessor.transform(features)
            print(data_scaled)
            predict = model.predict(data_scaled)
            return predict
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int,
                 ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_data_frame(self):
        try:
            custom_data_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            
            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)