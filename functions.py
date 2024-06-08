## Function used in the jupyter notebook
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

# Error handeling
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import datetime


def model_perform(model, model_name, params, X_train, y_train, X_test, y_test):

    gridsearch = GridSearchCV(model,
                             params,
                             scoring='neg_mean_squared_error',
                             cv=10,
                             verbose=0, n_jobs=-1)
    #list(gridsearch.get_params().keys())
    start = datetime.datetime.now()
    gridsearch.fit(X_train, y_train)
    print('\nBest hyperparameters:', gridsearch.best_params_)
    best_model = gridsearch.best_estimator_
    end = datetime.datetime.now()
    time = (end - start).microseconds
    #print(list(gridsearch.get_params().keys()))
    metrics_dataframe = calculate_metrics(best_model, model_name, X_train, y_train, X_test, y_test, time)
    return metrics_dataframe

metrics_dataframe = pd.DataFrame(columns = ['Model', 'R^2', 'MAPE', 'RMSE', 'Time'])
models = []
models_names = []
predictions_proba_list = []

def calculate_metrics(model, name, X_train, y_train, X_checked, y_checked, time):
    models.append(model)
    models_names.append(name)
    global metrics_dataframe
    predictions_train = model.predict(X_train)
    predictions = model.predict(X_checked)

    # Calculate R^2
    r = r2_score(y_train, predictions_train)

    # Calculate MAPE and RMSE
    MAPE = mean_absolute_percentage_error(y_checked, predictions)
    RMSE = np.sqrt(mean_squared_error(y_checked, predictions))

    metrics_dataframe = pd.concat([metrics_dataframe, pd.DataFrame.from_records([{'Model': name, 'R^2': r, 'MAPE': MAPE, 'RMSE': RMSE, 'Time' : time}])])

    return metrics_dataframe
