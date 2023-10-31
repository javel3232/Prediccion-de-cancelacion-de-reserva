import argparse
from loguru import logger
import os
import pandas as pd
import pickle

DEFAULT_INPUT_FILE = 'datasets/test.csv'
DEFAULT_MODEL_FILE = 'model.pkl'
DEFAULT_PREDICTIONS_FILE = 'predictions/predictions.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default=DEFAULT_INPUT_FILE, required=False, type=str, help='a csv file with input data (no targets)')
parser.add_argument('--model_file', default=DEFAULT_MODEL_FILE, required=False, type=str, help='a pkl file with a model already stored (see train.py)')
parser.add_argument('--predictions_file', default=DEFAULT_PREDICTIONS_FILE, required=False, type=str, help='a csv file where predictions will be saved to')

args = parser.parse_args()

input_file = args.input_file
model_file = args.model_file
predictions_file = args.predictions_file

if not os.path.isfile(input_file):
    logger.error(f"input file {input_file} does not exist")
    exit(-1)

if not os.path.isfile(model_file):
    logger.error(f"model file {model_file} does not exist")
    exit(-1)

logger.info("loading input data")
data = pd.read_csv(input_file)
train_data = data.drop('id', axis=1)

logger.info("loading model")
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

logger.info("making predictions")
predictions = model.predict(train_data)

logger.info(f"saving predictions to {predictions_file}")

submission = pd.DataFrame({'id': data['id'], 'target': predictions})
submission.to_csv(predictions_file, index=False)