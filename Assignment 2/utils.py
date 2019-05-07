import numpy as np
import pickle as p
import skimage
from scipy.signal import fftconvolve, convolve2d
import random
import os

cwd = os.getcwd()


def compare_two(list_a, list_b):
    result = []
    for a, b in zip(list_a, list_b):
        if a == b:
            result.append(a)

    return result


def load_models():
    path = os.path.join(cwd, "model")
    dirlist = os.listdir(path)
    models = []
    for file in dirlist:
        model_path = os.path.join(path, file)
        models.append(p.load(open(model_path, "rb")))

    return models


def load_predictions():
    path = os.path.join(cwd, "predictions")
    dirlist = os.listdir(path)
    pred = []
    for file in dirlist:
        file_path = os.path.join(path, file)
        pred.append(np.load(file_path))

    pred = np.array(pred)

    return pred


def load_data(x_folder):
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


def load_datav2(x_folder):
    dirlist = os.listdir(x_folder)
    x = []
    y = []
    for folder in dirlist:
        if folder == "test":
            continue
        classes = os.path.join(x_folder, folder)

        file_list = os.listdir(classes)

        for name in file_list:
            item = os.path.join(classes, name)
            item = os.path.join(item, "total.npy")
            x_vals = np.load(item)
            size, _, _, _, _ = x_vals.shape
            for i in x_vals:
                x.append(i.flatten())
                if folder == "confectionery":
                    y.append(1)
                else:
                    y.append(-1)

    x = np.array(x)
    y = np.array(y)

    return x, y

def label_to_num(y_labeled):
    prev = ''
    counter = 0
    y_dict = {}
    for item in y_labeled:
        if item != prev:
            prev = item
            y_dict[item] = counter
            counter += 1

    return np.array([y_dict[key] for key in y_labeled])


def create_test(path):
    save_path = os.path.join(path, "test")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dirlist = os.listdir(path)

    for folder in dirlist:
        # Path for 6973, 6574 class names
        classes = os.path.join(path, folder)

        file_list = os.listdir(classes)

        files = [name for name in file_list if name.endswith(".npy")]

        test_amount = len(files)//5

        print("Total files: " + str(len(files)))
        print("Test files: " + str(test_amount))

        chosen_samples = random.sample(files, k=test_amount)

        for item in chosen_samples:

            current_file_path = os.path.join(classes, item)

            save_path_class = os.path.join(save_path, folder)
            if not os.path.exists(save_path_class):
                os.mkdir(save_path_class)

            save_path_item = os.path.join(save_path_class, item)

            os.rename(current_file_path, save_path_item)


def convolve(im, rfs):

    rfs = rfs.swapaxes(2, 0).swapaxes(1, 2)

    feature_maps = []
    for filter in rfs:
        feature_b = convolve2d(im[:, :, 0], filter, mode='same', boundary='symm')
        feature_g = convolve2d(im[:, :, 1], filter, mode='same', boundary='symm')
        feature_r = convolve2d(im[:, :, 2], filter, mode='same', boundary='symm')
        feature_maps.append([feature_r, feature_g, feature_b])

    np_fm = np.array(feature_maps)
    return np_fm


def fft_convolve(im, rfs):

    rfs = rfs.swapaxes(2, 0).swapaxes(1, 2)

    feature_maps = []
    for filter in rfs:
        feature_b = fftconvolve(im[:, :, 0], filter, mode='same')
        feature_g = fftconvolve(im[:, :, 1], filter, mode='same')
        feature_r = fftconvolve(im[:, :, 2], filter, mode='same')
        feature_maps.append([feature_r, feature_g, feature_b])

    np_fm = np.array(feature_maps)
    return np_fm


def apply_relu(fm):
    result = np.maximum(0, fm)
    return result


def scikit_reduce(feature_maps, kernel_size):

    new_fm = np.zeros((38, 16, 16, 3))
    for ind, im in enumerate(feature_maps):
        new_fm[ind, :, :, 0] = skimage.measure.block_reduce(im[:, :, 0], kernel_size, np.max)
        new_fm[ind, :, :, 1] = skimage.measure.block_reduce(im[:, :, 1], kernel_size, np.max)
        new_fm[ind, :, :, 2] = skimage.measure.block_reduce(im[:, :, 2], kernel_size, np.max)

    return new_fm


def normalize(array):
    mini = array.min()
    array = np.subtract(array, mini)
    maxi = array.max()
    array = np.divide(array, maxi)
    return array
