import os
import numpy as np
import cv2
import copy
from scipy.signal import convolve2d
import skimage.measure
import random
from shutil import copyfile

cwd = os.getcwd()


def feature_extraction(im, rfs):

    rfs = rfs.swapaxes(2, 0).swapaxes(1, 2)

    feature_maps = []
    for filter in rfs:
        feature_b = convolve2d(im[:, :, 0], filter, mode='same', boundary='symm')
        feature_g = convolve2d(im[:, :, 1], filter, mode='same', boundary='symm')
        feature_r = convolve2d(im[:, :, 2], filter, mode='same', boundary='symm')
        feature_maps.append([feature_b, feature_g, feature_r])

    np_fm = np.array(feature_maps)
    np_fm = np_fm.swapaxes(1, 3).swapaxes(1, 2)
    return np_fm


def apply_relu(fm):
    result = np.maximum(0, fm)
    return result


def reduce_dimension(feature_maps, stride):
    new_fm = np.zeros((38, 16, 16, 3))
    for ind, im in enumerate(feature_maps):
        for i in range(16):
            for j in range(16):
                for c in range(3):
                    window = im[i*stride:(i*stride+stride), j*stride:j*stride+stride, c]
                    w_max = np.max(window)
                    new_fm[ind, i, j, c] = w_max

    return new_fm


def scikit_reduce(feature_maps, kernel_size):

    new_fm = np.zeros((38, 16, 16, 3))
    for ind, im in enumerate(feature_maps):
        new_fm[ind, :, :, 0] = skimage.measure.block_reduce(im[:, :, 0], kernel_size, np.max)
        new_fm[ind, :, :, 1] = skimage.measure.block_reduce(im[:, :, 1], kernel_size, np.max)
        new_fm[ind, :, :, 2] = skimage.measure.block_reduce(im[:, :, 2], kernel_size, np.max)

    return new_fm


def simplify_dataset(path):
    save_path = os.path.join(cwd, "mini")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dirlist = os.listdir(path)

    for folder in dirlist:
        # Path for 6973, 6574 class names
        classes = os.path.join(path, folder)

        file_list = os.listdir(classes)

        files = [name for name in file_list if name.endswith(".jpg")]

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

            copyfile(current_file_path, save_path_item)


if __name__ == "__main__":
    mainFolder = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\" \
                 "SKU_Recognition_Dataset\\SKU_Recognition_Dataset\\confectionery"

    simplify_dataset(mainFolder)

    '''
    rfs = np.load('rfsfilters.npy')
    ws = 8
    im = cv2.imread("example.jpg")
    fm = feature_extraction(im, rfs)
    fm_relu = apply_relu(fm)
    result = reduce_dimension(fm_relu, ws)
    result_np = np.array(result)
    np.save("example_result.npy", result_np)
    
    
    im = cv2.imread("example.jpg")
    rfs = np.load('rfsfilters.npy')


    ws = 8
    fm = feature_extraction(im, rfs)
    fm_relu = apply_relu(fm)
    result = reduce_dimension(fm_relu, ws)
    result2 = scikit_reduce(fm_relu, (ws, ws))

    if (result==result2).all():
        print("It was correct")
    else:
        print("It was wrong")
    '''


