import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, Y_train,X_test,Y_test,models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]
            
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, Y_train)
            
            print(gs.best_params_)
            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)
            
            Y_test_pred = model.predict(X_test)

            test_model_score = r2_score(Y_test, Y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        print(report)
        return report

    except Exception as e:
        raise CustomException(e, sys)