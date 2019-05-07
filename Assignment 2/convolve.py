from utils import fft_convolve, apply_relu, scikit_reduce
from scipy.io import loadmat
import numpy as np
import time
import cv2
import os

cwd = os.getcwd()

if __name__ == "__main__":
    mainFolder = ["C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\" \
                     "SKU_Recognition_Dataset\\SKU_Recognition_Dataset\\icecream"]

    rfs = loadmat('rfs.mat')
    rfs = rfs['rfs']

    ws = 8
    save_folder_a = os.path.join(cwd, str(ws)+"x"+str(ws))

    if not os.path.exists(save_folder_a):
        os.mkdir(save_folder_a)

    for f in mainFolder:
        prev_root = ""
        save_folder = os.path.join(save_folder_a, os.path.basename(os.path.normpath(f)))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        for root, dirs, files in os.walk(f):
            class_total = []
            file_list = [name for name in files if name.endswith(".jpg")]
            if len(files) != 0:
                im_class = os.path.basename(os.path.normpath(root))
                save_path = os.path.join(save_folder, im_class)
            i = 0
            for name in files:
                i += 1
                print(str(i) + "/" + str(len(files)))
                if not root == prev_root:
                    prev_root = root
                    print("We are in: " + im_class)
                if name.endswith(".jpg"):
                    im_path = os.path.join(root, name)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    save_im = os.path.join(save_path, name.replace(".jpg", ".npy"))
                    im = cv2.imread(im_path)
                    fm = fft_convolve(im, rfs)
                    fm_relu = apply_relu(fm)
                    result = scikit_reduce(fm_relu, (ws, ws))
                    class_total.append(result)
            if len(files) != 0:
                np.save(os.path.join(save_path, "total.npy"), class_total)



