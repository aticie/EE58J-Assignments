from sklearn.ensemble import AdaBoostClassifier
from utils import load_datav2, label_to_num
import os
import numpy as np
import pickle
import time

cwd = os.getcwd()

train_folder = os.path.join(cwd, "8x8")
test_folder = os.path.join(cwd, "test")
model_folder = os.path.join(cwd, "model")
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
prediction_folder = os.path.join(cwd, "predictions")
if not os.path.exists(prediction_folder):
    os.mkdir(prediction_folder)

x, y = load_datav2(train_folder, False)
x_test, y_test = load_datav2(test_folder, True)

iters = [100, 500, 1000, 5000]

cls = []
for i in iters:
    cls.append(AdaBoostClassifier(n_estimators=i))

y = y.flatten()

for idx, iter in enumerate(iters):
    if iter == 5000:
        continue
    start = time.time()
    c = cls[idx]
    c.fit(x, y)
    y_predict = c.predict(x_test)
    np.save(os.path.join(prediction_folder, "{}_iter.npy".format(iter)), y_predict)
    pickle.dump(c, open(os.path.join(model_folder, "cls{}.p".format(iter)), "wb"))

    end = time.time()

    total = end - start

    print("{} iterations is done, took ".format(iter) + str(total) + " secs")