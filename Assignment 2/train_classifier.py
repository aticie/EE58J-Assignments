from sklearn.ensemble import AdaBoostClassifier
import os
import numpy as np
import time

cwd = os.getcwd()

def load_train(x_folder):
    dirlist = os.listdir(x_folder)
    x = []
    y = []
    for folder in dirlist:
        if folder == "test":
            continue
        classes = os.path.join(x_folder, folder)

        file_list = os.listdir(classes)

        for name in file_list:
            x.append(np.load(os.path.join(classes, name)).flatten())
            y.append(folder)

    return x, y


def load_test(x_folder):
    dirlist = os.listdir(x_folder)
    x = []
    y = []
    for folder in dirlist:
        classes = os.path.join(x_folder, folder)

        file_list = os.listdir(classes)

        for name in file_list:
            x.append(np.load(os.path.join(classes, name)).flatten())
            y.append(folder)

    return x, y


train_folder = os.path.join(cwd, "8x8")
test_folder = os.path.join(train_folder, "test")
x, y = load_train(train_folder)
x_test, y_test = load_test(test_folder)

#setattr(AdaBoostClassifier, "n_classes", 20)

cls100 = AdaBoostClassifier(n_estimators=100)
cls500 = AdaBoostClassifier(n_estimators=500)
cls1000 = AdaBoostClassifier(n_estimators=1000)
cls5000 = AdaBoostClassifier(n_estimators=5000)

start = time.time()

cls100.fit(x, y)
y_predict100 = cls100.predict(x_test)
np.save("y_predict100.npy", y_predict100)
np.save("cls100.npy", cls100)

end = time.time()

total = end - start

print("100 iterations is done, took "+ str(total) + " secs")

start = time.time()

cls500.fit(x, y)
y_predict500 = cls500.predict(x_test)
np.save("y_predict500.npy", y_predict500)
np.save("cls500.npy", cls500)

end = time.time()

total = end - start

print("500 iterations is done, took "+ str(total) + " secs")

start = time.time()

cls1000.fit(x, y)
y_predict1000 = cls1000.predict(x_test)
np.save("y_predict1000.npy", y_predict1000)
np.save("cls1000.npy", cls1000)

end = time.time()

total = end - start

print("1000 iterations is done, took "+ str(total) + " secs")

start = time.time()

cls5000.fit(x, y)
y_predict5000 = cls5000.predict(x_test)
np.save("y_predict5000.npy", y_predict5000)
np.save("cls5000.npy", cls5000)

end = time.time()

total = end - start

print("5000 iterations is done, took "+ str(total) + " secs")