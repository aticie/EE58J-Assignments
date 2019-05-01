import os
import numpy as np
import cv2
from scipy.signal import convolve2d
import skimage.measure

cwd = os.getcwd()


def feature_extraction(im, rfs):

    rfs = rfs.swapaxes(2, 0).swapaxes(1, 2)

    feature_maps = []
    for filter in rfs:
        feature_b = convolve2d(im[:, :, 0], filter, mode='same', boundary='symm')
        feature_g = convolve2d(im[:, :, 1], filter, mode='same', boundary='symm')
        feature_r = convolve2d(im[:, :, 2], filter, mode='same', boundary='symm')
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
    min = array.min()
    array = np.subtract(array, min)
    max = array.max()
    array = np.divide(array, max)
    return array


if __name__ == "__main__":
    mainFolder = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\" \
                 "SKU_Recognition_Dataset\\SKU_Recognition_Dataset\\confectionery"
    rfs = np.load('rfsfilters.npy')
    ws = 8
    save_folder = os.path.join(cwd, str(ws)+"x"+str(ws))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    prev_root = ""
    for root, dirs, files in os.walk(mainFolder):
        file_list = [name for name in files if name.endswith(".jpg")]
        i = 0
        for name in files:
            i += 1
            print(str(i)+"/"+str(len(files)))
            im_class = os.path.basename(os.path.normpath(root))
            if not root == prev_root:
                prev_root = root
                print("We are in: " + im_class)
            if name.endswith(".jpg"):
                print("Extracting features from: "+name)
                im_path = os.path.join(root, name)
                save_path = os.path.join(save_folder, im_class)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_im = os.path.join(save_path, name.replace(".jpg", ".npy"))
                im = cv2.imread(im_path)
                fm = feature_extraction(im, rfs)
                fm_relu = apply_relu(fm)
                result = scikit_reduce(fm_relu, (ws, ws))
                np.save(save_im, result)







