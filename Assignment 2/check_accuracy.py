import os
import numpy as np
from utils import load_predictions, load_models, load_data, compare_two, label_to_num
import pickle

cwd = os.getcwd()

data_path = os.path.join(cwd, "8x8")

predictions = load_predictions()
models = load_models()

x_train, y_train = load_data(data_path)
test_path = os.path.join(os.path.join(cwd, "8x8"), "test")
x_test, y_test = load_data(test_path)

y_train = label_to_num(y_train)
y_test = label_to_num(y_test)

error = []

for pred in predictions:
    error.append(compare_two(y_test, pred))

print(len(error[0]))