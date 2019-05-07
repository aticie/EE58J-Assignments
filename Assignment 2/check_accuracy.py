import os
import numpy as np
from utils import load_predictions, load_models, load_datav2, compare_two, label_to_num
import pickle

cwd = os.getcwd()

data_path = os.path.join(cwd, "8x8")

predictions = load_predictions()
models = load_models()

x_train, y_train = load_datav2(data_path)

y_train = label_to_num(y_train)

error = []

for pred in predictions:
    error.append(compare_two(y_train, pred))

print(len(error[0]))