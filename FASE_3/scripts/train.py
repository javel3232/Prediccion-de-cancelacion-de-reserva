
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import os

def train_and_save_model(data_file='datasets/train.csv', model_file='model.pkl', overwrite_model=True):
    if os.path.isfile(model_file):
        if overwrite_model:
            logger.info(f"Overwriting existing model file {model_file}")
        else:
            logger.info(f"Model file {model_file} exists. Exiting. Use --overwrite_model option.")
            return None

    # Load data from a CSV file
    logger.info("Loading train data")
    data = pd.read_csv(data_file)

    # Data cleaning
    train_data = data.drop('id', axis=1)

    # split data into X and y
    X = train_data.drop(['target'], axis=1)
    y = train_data['target']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=7)

    # fit model on training data
    logger.info("Fitting model")
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))

    # save the trained model
    logger.info(f"Saving model to {model_file}")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    return model
