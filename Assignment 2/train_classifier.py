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

cls100 = AdaBoostClassifier(n_estimators=100)

start = time.time()
cls100.fit(x, y)
y_predict100 = cls100.predict(x_test)
y_train100 = cls100.predict(x)
np.save(os.path.join(prediction_folder, "100_iter.npy"), y_predict100)
pickle.dump(cls100, open(os.path.join(model_folder, "cls100.p"), "wb"))

end = time.time()

total = end - start

print("100 iterations is done, took " + str(total) + " secs")