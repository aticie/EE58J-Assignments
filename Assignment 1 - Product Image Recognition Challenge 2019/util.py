import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()


def resizeBatch(inPath):
    prevRoot = ""
    for root, dirs, files in os.walk(inPath, topdown=False):
        for name in files:
            if not root == prevRoot:
                prevRoot = root
                print("We are in: " + os.path.basename(os.path.normpath(root)))

            if not name.endswith(".jpg"):
                print(name + " - Doesn't end with .jpg")
                continue

            filePath = os.path.join(root, name)
            # Read image in RGB
            im = cv2.imread(filePath, 1)
            # If image is already 128x128, do nothing
            if im.shape == (128, 128, 3):
                # print(name+" is already resized!")
                continue
            '''
            ----DEBUG----
            # Resize image to 128x128
            # print("Resizing: "+name)
            # print("Dimensions: "+im.shape)
            ----DEBUG----
            '''
            newimg = cv2.resize(im, (128, 128))
            # Overwrite resized image
            cv2.imwrite(filePath, newimg)


def colorHist(path, windowNr, bin_num):
    prevRoot = ""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                continue
            if not root == prevRoot:
                prevRoot = root
                print("We are in: " + os.path.basename(os.path.normpath(root)))

            if name.endswith(".jpg"):
                filePath = os.path.join(root, name)
                im = cv2.imread(filePath, 1)
                HSV_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

                stride = int(128 / windowNr)
                full_hsv_hist = []
                for i in range(windowNr):
                    for j in range(windowNr):
                        window = im[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]
                        '''
                        ----DEBUG----
                        print(window[:,:,0].shape)
                        cv2.imshow("Window",window)
                        cv2.waitKey(0)
                        ----DEBUG----
                        '''
                        hHist = np.histogram(window[:, :, 0], bins=bin_num)
                        sHist = np.histogram(window[:, :, 1], bins=bin_num)
                        vHist = np.histogram(window[:, :, 2], bins=bin_num)
                        hNorm = np.divide(hHist[0], np.sum(hHist[0]))
                        sNorm = np.divide(sHist[0], np.sum(sHist[0]))
                        vNorm = np.divide(vHist[0], np.sum(vHist[0]))
                        hsvHist = np.concatenate((hNorm, sNorm, vNorm))
                        full_hsv_hist.append(hsvHist)
                        '''
                        ----DEBUG----
                        print(hNorm)
                        plt.hist(window[:,:,0].flatten(), bins=10)
                        plt.title("Hue Histogram")
                        plt.show()
                        cv2.imshow("Window",window)
                        cv2.waitKey(0)
                        ----DEBUG----
                        '''

                saveName = name.replace(".jpg", "_color.npy")
                np.save(os.path.join(root, saveName), full_hsv_hist)

    return 0


def HOGHist(path, windowNr, bin_num):
    prevRoot = ""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                continue
            if not root == prevRoot:
                prevRoot = root
                print("We are in: " + os.path.basename(os.path.normpath(root)))
            if name.endswith(".jpg"):
                filePath = os.path.join(root, name)
                im = cv2.imread(filePath, 0)
                sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)

                orient = np.arctan2(sobely, sobelx) * 180 / np.pi
                full_orient = []
                stride = int(128 / windowNr)
                for i in range(windowNr):
                    for j in range(windowNr):
                        window = orient[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]
                        orient_hist = np.histogram(window, bins=bin_num)
                        orient_norm = np.divide(orient_hist[0], np.sum(orient_hist[0]))
                        full_orient.append(orient_norm)

                saveName = name.replace(".jpg", "_orient.npy")
                np.save(os.path.join(root, saveName), full_orient)

    return 0


if __name__ == "__main__":
    inPath = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\confectionery"
    resizeBatch(inPath)
