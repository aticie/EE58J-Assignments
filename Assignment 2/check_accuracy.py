import os
import numpy as np
from utils import load_predictions, load_models, load_datav2, compare_two, label_to_num
import pickle

cwd = os.getcwd()

data_path = os.path.join(cwd, "8x8")
test_path = os.path.join(cwd, "test")

predictions = load_predictions()
models = load_models()

x_train, y_train = load_datav2(data_path, False)
x_test, y_test = load_datav2(test_path, True)

error = []
confMat = []

for pred in predictions:
    single_error, single_confMat = compare_two(y_test, pred)
    error.append(single_error)
    confMat.append(single_confMat)

pred1000 = error[0]
pred100 = error[1]
pred500 = error[2]
print(len(pred100))
print(len(pred500))
print(len(pred1000))
print(len(y_test))
print(confMat[1])
print(confMat[2])
print(confMat[0])