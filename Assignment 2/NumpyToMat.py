import numpy as np
import scipy.io
import os

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

    x = np.array(x)
    y = np.array(y)

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

scipy.io.savemat('train.mat', dict(x=x, y=y))
scipy.io.savemat('test.mat', dict(x=x_test, y=y_test))
