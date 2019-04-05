import os
import numpy as np
import cv2
import copy
from scipy.signal import convolve2d

cwd = os.getcwd()


def feature_extraction(im, rfs):

    rfs = rfs.swapaxes(2, 0).swapaxes(1, 2)

    feature_maps = []
    for filter in rfs:
        feature_b = convolve2d(im[:, :, 0], filter, mode='same', boundary='symm')
        feature_g = convolve2d(im[:, :, 1], filter, mode='same', boundary='symm')
        feature_r = convolve2d(im[:, :, 2], filter, mode='same', boundary='symm')
        feature_maps.append([feature_r, feature_g, feature_b])

    return feature_maps


def apply_relu(fm):
    result = np.maximum(0, fm)
    return result


def reduce_dimension(feature_maps, stride):
    imsize = 128
    new_fm = []
    new_fm_w = np.zeros((int(128/stride), int(128/stride)))
    for im in feature_maps:
        for channel in im:
            for i in range(0, imsize, stride):
                for j in range(0, imsize, stride):
                    window = channel[j:(j+1)*stride, i:(i+1)*stride]
                    w_max = np.max(window)
                    new_fm_w[int(j/stride), int(i/stride)] = w_max

            new_fm.append(copy.deepcopy(new_fm_w))

    return new_fm


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
        for name in files:
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
                result = reduce_dimension(fm_relu, ws)
                np.save(save_im, result)







