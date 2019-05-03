from sklearn.ensemble import AdaBoostClassifier
from utils import load_data, label_to_num
import os
import numpy as np
import pickle
import time

cwd = os.getcwd()

train_folder = os.path.join(cwd, "8x8")
test_folder = os.path.join(train_folder, "test")
model_folder = os.path.join(cwd, "model")
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
prediction_folder = os.path.join(cwd, "predictions")
if not os.path.exists(prediction_folder):
    os.mkdir(prediction_folder)
x, y = load_data(train_folder)
y = label_to_num(y)
x_test, y_test = load_data(test_folder)
y_test = label_to_num(y_test)
#setattr(AdaBoostClassifier, "n_classes", 20)

iters = [1000, 5000]

for iter in iters:
    c = AdaBoostClassifier(n_estimators=iter)
    start = time.time()
    c.fit(x, y)
    y_predict100 = c.predict(x_test)
    np.save(os.path.join(prediction_folder, "{}_iter.npy".format(iter)), y_predict100)
    pickle.dump(c, open(os.path.join(model_folder, "cls{}.p".format(iter)), "wb"))

    end = time.time()

    total = end - start

    print("{} iterations is done, took ".format(iter) + str(total) + " secs")


