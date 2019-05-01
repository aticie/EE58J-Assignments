import os
import numpy as np

cwd = os.getcwd()

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


def compare_two(list_a, list_b):
    result = []
    for a, b in zip(list_a, list_b):
        if a == b:
            result.append(a)

    return result

ypredict100 = np.load("y_predict100.npy")
ypredict500 = np.load("y_predict500.npy")
ypredict1000 = np.load("y_predict1000.npy")
ypredict5000 = np.load("y_predict5000.npy")

test_path = os.path.join(os.path.join(cwd, "8x8mini"), "test")
x_test, y_test = load_test(test_path)

result100 = compare_two(y_test, ypredict100)
result500 = compare_two(y_test, ypredict500)
result1000 = compare_two(y_test, ypredict1000)
result5000 = compare_two(y_test, ypredict5000)

print(len(result100))
print(len(result500))
print(len(result1000))
print(len(result5000))
print(len(y_test))