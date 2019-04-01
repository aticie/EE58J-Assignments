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
                        window = HSV_im[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]
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


def nnOrient(data_path):

    print("N-nearest Neighbour classification for gradient orientation is starting!")

    print("----- Training Phase -----")

    if os.path.exists("orient_pretrained_model_x.npy") and os.path.exists("orient_pretrained_model_y.npy"):
        print("Pretrained model exists! Using it!")
        train_x = np.load("orient_pretrained_model_x.npy")
        train_y = np.load("orient_pretrained_model_y.npy")
    else:
        print("Pretrained model doesn't exist, creating it...")
        prev_root = ""
        train_x = []
        train_y = []
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                if name == ".DS_Store":
                    continue
                if not root == prev_root:
                    prev_root = root
                    #print("We are in: " + os.path.basename(os.path.normpath(root)))
                two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
                if two_up == "test":
                    continue
                if not name.endswith(".npy"):
                    continue
                if name.endswith("_orient.npy"):
                    file_path = os.path.join(root, name)
                    train_x.append(np.load(file_path))
                    train_y.append(os.path.basename(os.path.normpath(root)))

        np.save("orient_pretrained_model_x",train_x)
        np.save("orient_pretrained_model_y",train_y)

    conf_test_amnt = 0
    ice_test_amnt = 0
    laun_test_amnt = 0
    soft1_test_amnt = 0
    soft2_test_amnt = 0
    conf_correct_count = 0
    ice_correct_count = 0
    laun_correct_count = 0
    soft1_correct_count = 0
    soft2_correct_count = 0
    test_amnt = 0
    correct_count = 0
    prev_root = ""
    correctness_dict = dict()
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                continue
            if not name.endswith(".npy"):
                continue
            if not root == prev_root:
                prev_root = root
                #print("We are in: " + os.path.basename(os.path.normpath(root)))
            two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
            one_up = os.path.basename(os.path.abspath(os.path.join(root, "..")))
            if two_up == "test":
                if name.endswith("_orient.npy"):
                    file_path = os.path.join(root, name)
                    test_x = np.load(file_path)
                    test_real = os.path.basename(os.path.normpath(root))
                    min_mean = np.Inf
                    right_index = -1
                    '''
                    for index, item in enumerate(train_x):
                        current_mean = np.mean(np.abs(np.subtract(test_x, item)))
                        if min_mean > current_mean:
                            min_mean = current_mean
                            right_index = index
                    '''
                    train_x = np.nan_to_num(train_x)
                    test_x = np.nan_to_num(test_x)
                    mean_vector = np.mean(np.abs(test_x-train_x), axis=2)
                    try:
                        mean_vector = np.sum(mean_vector, axis=1)
                    except:
                        mean_vector = mean_vector
                    min_mean_ind = np.argmin(mean_vector)
                    if test_real == train_y[min_mean_ind]:
                        if test_real in correctness_dict:
                            correctness_dict[test_real]['correct'] += 1
                            correctness_dict[test_real]['total'] += 1
                        else:
                            correctness_dict[test_real] = {'correct': 1, 'total': 1}
                        #print("Correct!")
                        if one_up == "confectionery":
                            conf_correct_count += 1
                            conf_test_amnt += 1
                        if one_up == "icecream":
                            ice_correct_count += 1
                            ice_test_amnt += 1
                        if one_up == "laundry":
                            laun_correct_count += 1
                            laun_test_amnt += 1
                        if one_up == "softdrinks-I":
                            soft1_correct_count += 1
                            soft1_test_amnt += 1
                        if one_up == "softdrinks-II":
                            soft2_correct_count += 1
                            soft2_test_amnt += 1
                        test_amnt += 1
                        correct_count += 1
                        acc = correct_count/test_amnt
                        #print("Accuracy :" + str(acc*100) + "\%")
                        if not test_amnt%250:
                            print("Accuracy :" + str(acc * 100) + "%")
                    else:
                        if test_real in correctness_dict:
                            correctness_dict[test_real]['total'] += 1
                        else:
                            correctness_dict[test_real] = {'correct': 0, 'total': 1}
                        #print("False!")
                        #print("Real class was: " + test_real)
                        #print("NN guessed: " + train_y[right_index])
                        #print("Min-mean was: " + str(min_mean))
                        if one_up == "confectionery":
                            conf_test_amnt += 1
                        if one_up == "icecream":
                            ice_test_amnt += 1
                        if one_up == "laundry":
                            laun_test_amnt += 1
                        if one_up == "softdrinks-I":
                            soft1_test_amnt += 1
                        if one_up == "softdrinks-II":
                            soft2_test_amnt += 1
                        test_amnt += 1
                        acc = correct_count / test_amnt
                        #print("Accuracy :" + str(acc * 100) + "\%")
                        if not test_amnt%250:
                            print("Accuracy :" + str(acc * 100) + "%")

    min_class_acc = 100
    min_class = ""
    min_c = 0
    min_t = 0
    for key, value in correctness_dict.items():
        class_acc = value['correct'] / value['total']
        if class_acc < min_class_acc:
            min_class = key
            min_class_acc = class_acc
            min_c = value['correct']
            min_t = value['total']

    print("Min acc class: ", min_class, " with acc: ", min_class_acc, " - ", min_c, "/", min_t)

    acc = correct_count / test_amnt
    conf_acc = conf_correct_count / conf_test_amnt
    ice_acc = ice_correct_count / ice_test_amnt
    laun_acc = laun_correct_count / laun_test_amnt
    soft1_acc = soft1_correct_count / soft1_test_amnt
    soft2_acc = soft2_correct_count / soft2_test_amnt
    print(str(correct_count) + " out of " + str(test_amnt) + " was correct.")
    print("Accuracy :" + str(acc))

    print("Confectionery - " + str(conf_correct_count) + " out of " + str(conf_test_amnt) + " was correct.")
    print("Accuracy :" + str(conf_acc))
    print("Icecream - " + str(ice_correct_count) + " out of " + str(ice_test_amnt) + " was correct.")
    print("Accuracy :" + str(ice_acc))
    print("Laundry - " + str(laun_correct_count) + " out of " + str(laun_test_amnt) + " was correct.")
    print("Accuracy :" + str(laun_acc))
    print("SoftDrinks-1 - " + str(soft1_correct_count) + " out of " + str(soft1_test_amnt) + " was correct.")
    print("Accuracy :" + str(soft1_acc))
    print("SoftDrinks-2 - " + str(soft2_correct_count) + " out of " + str(soft2_test_amnt) + " was correct.")
    print("Accuracy :" + str(soft2_acc))

    return 0


def nnColor(data_path):
    print("N-nearest Neighbour classification for color histograms is starting!")

    print("----- Training Phase -----")

    if os.path.exists("color_pretrained_model_x.npy") and os.path.exists("color_pretrained_model_y.npy"):
        print("Pretrained model exists! Using it!")
        train_x = np.load("color_pretrained_model_x.npy")
        train_y = np.load("color_pretrained_model_y.npy")
    else:
        prev_root = ""
        train_x = []
        train_y = []
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                if name == ".DS_Store":
                    continue
                if not root == prev_root:
                    prev_root = root
                    # print("We are in: " + os.path.basename(os.path.normpath(root)))
                two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
                if two_up == "test":
                    continue
                if not name.endswith(".npy"):
                    continue
                if name.endswith("_color.npy"):
                    file_path = os.path.join(root, name)
                    train_x.append(np.load(file_path))
                    train_y.append(os.path.basename(os.path.normpath(root)))

        np.save("color_pretrained_model_x", train_x)
        np.save("color_pretrained_model_y", train_y)

    conf_test_amnt = 0
    ice_test_amnt = 0
    laun_test_amnt = 0
    soft1_test_amnt = 0
    soft2_test_amnt = 0
    conf_correct_count = 0
    ice_correct_count = 0
    laun_correct_count = 0
    soft1_correct_count = 0
    soft2_correct_count = 0
    test_amnt = 0
    correct_count = 0
    prev_root = ""
    correctness_dict = dict()
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                continue
            if not name.endswith(".npy"):
                continue
            if not root == prev_root:
                prev_root = root
                # print("We are in: " + os.path.basename(os.path.normpath(root)))
            two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
            one_up = os.path.basename(os.path.abspath(os.path.join(root, "..")))
            if two_up == "test":
                if name.endswith("_color.npy"):
                    file_path = os.path.join(root, name)
                    test_x = np.load(file_path)
                    test_real = os.path.basename(os.path.normpath(root))
                    min_mean = np.Inf
                    right_index = -1
                    '''
                    for index, item in enumerate(train_x):
                        current_mean = np.mean(np.abs(np.subtract(test_x, item)))
                        if min_mean > current_mean:
                            min_mean = current_mean
                            right_index = index
                    '''
                    train_x = np.nan_to_num(train_x)
                    test_x = np.nan_to_num(test_x)

                    mean_vector = np.mean(np.abs(test_x - train_x), axis=2)
                    try:
                        mean_vector = np.sum(mean_vector, axis=1)
                    except:
                        mean_vector = mean_vector
                    min_mean_ind = np.argmin(mean_vector)
                    if test_real == train_y[min_mean_ind]:
                        # print("Correct!")
                        if test_real in correctness_dict:
                            correctness_dict[test_real]['correct'] += 1
                            correctness_dict[test_real]['total'] += 1
                        else:
                            correctness_dict[test_real] = {'correct': 1, 'total': 1}
                        if one_up == "confectionery":
                            conf_correct_count += 1
                            conf_test_amnt += 1
                        if one_up == "icecream":
                            ice_correct_count += 1
                            ice_test_amnt += 1
                        if one_up == "laundry":
                            laun_correct_count += 1
                            laun_test_amnt += 1
                        if one_up == "softdrinks-I":
                            soft1_correct_count += 1
                            soft1_test_amnt += 1
                        if one_up == "softdrinks-II":
                            soft2_correct_count += 1
                            soft2_test_amnt += 1
                        test_amnt += 1
                        correct_count += 1
                        acc = correct_count / test_amnt
                        # print("Accuracy :" + str(acc*100) + "\%")
                        if not test_amnt % 250:
                            print("Accuracy :" + str(acc * 100) + "%")
                    else:
                        # print("False!")
                        # print("Real class was: " + test_real)
                        # print("NN guessed: " + train_y[right_index])
                        # print("Min-mean was: " + str(min_mean))
                        if test_real in correctness_dict:
                            correctness_dict[test_real]['total'] += 1
                        else:
                            correctness_dict[test_real] = {'correct': 0, 'total': 1}
                        if one_up == "confectionery":
                            conf_test_amnt += 1
                        if one_up == "icecream":
                            ice_test_amnt += 1
                        if one_up == "laundry":
                            laun_test_amnt += 1
                        if one_up == "softdrinks-I":
                            soft1_test_amnt += 1
                        if one_up == "softdrinks-II":
                            soft2_test_amnt += 1
                        test_amnt += 1
                        acc = correct_count / test_amnt
                        # print("Accuracy :" + str(acc * 100) + "\%")
                        if not test_amnt % 250:
                            print("Accuracy :" + str(acc * 100) + "%")

    min_class_acc = 100
    min_class = ""
    min_c = 0
    min_t = 0
    for key, value in correctness_dict.items():
        class_acc = value['correct']/value['total']
        if class_acc < min_class_acc:
            min_class = key
            min_class_acc = class_acc
            min_c = value['correct']
            min_t = value['total']

    print("Min acc class: ", min_class, " with acc: ", min_class_acc, " - ", min_c, "/", min_t)

    acc = correct_count / test_amnt
    conf_acc = conf_correct_count / conf_test_amnt
    ice_acc = ice_correct_count / ice_test_amnt
    laun_acc = laun_correct_count / laun_test_amnt
    soft1_acc = soft1_correct_count / soft1_test_amnt
    soft2_acc = soft2_correct_count / soft2_test_amnt
    print(str(correct_count) + " out of " + str(test_amnt) + " was correct.")
    print("Accuracy :" + str(acc))

    print("Confectionery - " + str(conf_correct_count) + " out of " + str(conf_test_amnt) + " was correct.")
    print("Accuracy :" + str(conf_acc))
    print("Icecream - " + str(ice_correct_count) + " out of " + str(ice_test_amnt) + " was correct.")
    print("Accuracy :" + str(ice_acc))
    print("Laundry - " + str(laun_correct_count) + " out of " + str(laun_test_amnt) + " was correct.")
    print("Accuracy :" + str(laun_acc))
    print("SoftDrinks-1 - " + str(soft1_correct_count) + " out of " + str(soft1_test_amnt) + " was correct.")
    print("Accuracy :" + str(soft1_acc))
    print("SoftDrinks-2 - " + str(soft2_correct_count) + " out of " + str(soft2_test_amnt) + " was correct.")
    print("Accuracy :" + str(soft2_acc))

    return 0


def nnCombine(data_path):

    print("N-nearest Neighbour classification for combination of both is starting!")
    print("----- Training Phase -----")

    if os.path.exists("orient_pretrained_model_x.npy") and os.path.exists("orient_pretrained_model_y.npy"):
        print("Pretrained model for orientation exists! Using it!")
        or_train_x = np.load("orient_pretrained_model_x.npy")
        or_train_y = np.load("orient_pretrained_model_y.npy")
    else:
        print("Pretrained model doesn't exist, creating it...")
        prev_root = ""
        or_train_x = []
        or_train_y = []
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                if name == ".DS_Store":
                    continue
                if not root == prev_root:
                    prev_root = root
                    # print("We are in: " + os.path.basename(os.path.normpath(root)))
                two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
                if two_up == "test":
                    continue
                if not name.endswith(".npy"):
                    continue
                if name.endswith("_orient.npy"):
                    file_path = os.path.join(root, name)
                    or_train_x.append(np.load(file_path))
                    or_train_y.append(os.path.basename(os.path.normpath(root)))

        np.save("orient_pretrained_model_x", or_train_x)
        np.save("orient_pretrained_model_y", or_train_y)
    if os.path.exists("color_pretrained_model_x.npy") and os.path.exists("color_pretrained_model_y.npy"):
        print("Pretrained model for color exists! Using it!")
        cl_train_x = np.load("color_pretrained_model_x.npy")
        cl_train_y = np.load("color_pretrained_model_y.npy")
    else:
        prev_root = ""
        cl_train_x = []
        cl_train_y = []
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                if name == ".DS_Store":
                    continue
                if not root == prev_root:
                    prev_root = root
                    # print("We are in: " + os.path.basename(os.path.normpath(root)))
                two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
                if two_up == "test":
                    continue
                if not name.endswith(".npy"):
                    continue
                if name.endswith("_color.npy"):
                    file_path = os.path.join(root, name)
                    cl_train_x.append(np.load(file_path))
                    cl_train_y.append(os.path.basename(os.path.normpath(root)))

        np.save("color_pretrained_model_x", cl_train_x)
        np.save("color_pretrained_model_y", cl_train_y)

    conf_test_amnt = 0
    ice_test_amnt = 0
    laun_test_amnt = 0
    soft1_test_amnt = 0
    soft2_test_amnt = 0
    conf_correct_count = 0
    ice_correct_count = 0
    laun_correct_count = 0
    soft1_correct_count = 0
    soft2_correct_count = 0
    correctness_dict = dict()
    test_amnt = 0
    correct_count = 0
    prev_root = ""
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                continue
            if not name.endswith(".npy"):
                continue
            if not root == prev_root:
                prev_root = root
                # print("We are in: " + os.path.basename(os.path.normpath(root)))
            two_up = os.path.basename(os.path.abspath(os.path.join(root, "../..")))
            one_up = os.path.basename(os.path.abspath(os.path.join(root, "..")))
            if two_up == "test":
                if name.endswith("_orient.npy"):
                    file_path_or = os.path.join(root, name)
                    cl_name = name.replace("_orient.npy","_color.npy")
                    file_path_cl = os.path.join(root, cl_name)

                    or_test_x = np.load(file_path_or)
                    cl_test_x = np.load(file_path_cl)
                    test_real = os.path.basename(os.path.normpath(root))
                    min_mean = np.Inf
                    right_index = -1
                    '''
                    for index, item in enumerate(train_x):
                        current_mean = np.mean(np.abs(np.subtract(test_x, item)))
                        if min_mean > current_mean:
                            min_mean = current_mean
                            right_index = index
                    '''
                    or_train_x = np.nan_to_num(or_train_x)
                    or_test_x = np.nan_to_num(or_test_x)
                    cl_train_x = np.nan_to_num(cl_train_x)
                    cl_test_x = np.nan_to_num(cl_test_x)
                    cl_mean_vector = np.mean(np.abs(cl_test_x - cl_train_x), axis=2)
                    or_mean_vector = np.mean(np.abs(or_test_x - or_train_x), axis=2)
                    mean_vector = or_mean_vector + cl_mean_vector
                    try:
                        mean_vector = np.sum(mean_vector, axis=1)
                    except:
                        mean_vector = mean_vector
                    min_mean_ind = np.argmin(mean_vector)
                    if test_real == cl_train_y[min_mean_ind]:
                        if test_real in correctness_dict:
                            correctness_dict[test_real]['correct'] += 1
                            correctness_dict[test_real]['total'] += 1
                        else:
                            correctness_dict[test_real] = {'correct': 1, 'total': 1}
                        if one_up == "confectionery":
                            conf_correct_count += 1
                            conf_test_amnt += 1
                        if one_up == "icecream":
                            ice_correct_count += 1
                            ice_test_amnt += 1
                        if one_up == "laundry":
                            laun_correct_count += 1
                            laun_test_amnt += 1
                        if one_up == "softdrinks-I":
                            soft1_correct_count += 1
                            soft1_test_amnt += 1
                        if one_up == "softdrinks-II":
                            soft2_correct_count += 1
                            soft2_test_amnt += 1
                        # print("Correct!")
                        test_amnt += 1
                        correct_count += 1
                        acc = correct_count / test_amnt
                        # print("Accuracy :" + str(acc*100) + "\%")
                        if not test_amnt % 250:
                            print("Accuracy :" + str(acc * 100) + "%")
                    else:
                        # print("False!")
                        # print("Real class was: " + test_real)
                        # print("NN guessed: " + train_y[right_index])
                        # print("Min-mean was: " + str(min_mean))
                        if test_real in correctness_dict:
                            correctness_dict[test_real]['total'] += 1
                        else:
                            correctness_dict[test_real] = {'correct': 0, 'total': 1}
                        if one_up == "confectionery":
                            conf_test_amnt += 1
                        if one_up == "icecream":
                            ice_test_amnt += 1
                        if one_up == "laundry":
                            laun_test_amnt += 1
                        if one_up == "softdrinks-I":
                            soft1_test_amnt += 1
                        if one_up == "softdrinks-II":
                            soft2_test_amnt += 1
                        test_amnt += 1
                        acc = correct_count / test_amnt
                        # print("Accuracy :" + str(acc * 100) + "\%")
                        if not test_amnt % 250:
                            print("Accuracy :" + str(acc * 100) + "%")

    min_class_acc = 100
    min_class = ""
    min_c = 0
    min_t = 0
    for key, value in correctness_dict.items():
        class_acc = value['correct'] / value['total']
        if class_acc < min_class_acc:
            min_class = key
            min_class_acc = class_acc
            min_c = value['correct']
            min_t = value['total']

    print("Min acc class: ", min_class, " with acc: ", min_class_acc, " - ", min_c, "/", min_t)

    acc = correct_count / test_amnt
    conf_acc = conf_correct_count / conf_test_amnt
    ice_acc = ice_correct_count / ice_test_amnt
    laun_acc = laun_correct_count / laun_test_amnt
    soft1_acc = soft1_correct_count / soft1_test_amnt
    soft2_acc = soft2_correct_count / soft2_test_amnt
    print(str(correct_count) + " out of " + str(test_amnt) + " was correct.")
    print("Accuracy :" + str(acc))

    print("Confectionery - " + str(conf_correct_count) + " out of " + str(conf_test_amnt) + " was correct.")
    print("Accuracy :" + str(conf_acc))
    print("Icecream - " + str(ice_correct_count) + " out of " + str(ice_test_amnt) + " was correct.")
    print("Accuracy :" + str(ice_acc))
    print("Laundry - " + str(laun_correct_count) + " out of " + str(laun_test_amnt) + " was correct.")
    print("Accuracy :" + str(laun_acc))
    print("SoftDrinks-1 - " + str(soft1_correct_count) + " out of " + str(soft1_test_amnt) + " was correct.")
    print("Accuracy :" + str(soft1_acc))
    print("SoftDrinks-2 - " + str(soft2_correct_count) + " out of " + str(soft2_test_amnt) + " was correct.")
    print("Accuracy :" + str(soft2_acc))

    return 0


if __name__ == "__main__":
    inPath = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\confectionery"
    resizeBatch(inPath)
