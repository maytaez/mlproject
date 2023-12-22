# utils will contain all the common things/functionalities which the entire project can use
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        # going through each and every model
        for i in range(len(list(models))):
            model = list(models.values())[i]
            # going through each and every params
            para = param[list(models.keys())[i]]

            # Using GridSearchCv as an estimator as it considers all the parameters in combinations
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # after selecting the best parameters and us the best parameters as required
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # appending the model score(r2 score) inside report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


# function to read the pickle file and loading the file using dill.
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
