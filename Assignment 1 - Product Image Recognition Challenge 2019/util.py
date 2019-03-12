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
                print("We are in: "+ os.path.basename(os.path.normpath(root)))
            
            if not name.endswith(".jpg"):
                print(name+" - Doesn't end with .jpg")
                continue
            
            filePath = os.path.join(root,name)
            # Read image in RGB
            im = cv2.imread(filePath,1)
            # If image is already 128x128, do nothing
            if im.shape == (128,128,3):
                #print(name+" is already resized!")
                continue
            '''
            ----DEBUG----
            # Resize image to 128x128
            # print("Resizing: "+name)
            # print("Dimensions: "+im.shape)
            ----DEBUG----
            '''
            newimg = cv2.resize(im,(128,128))
            # Overwrite resized image
            cv2.imwrite(filePath,newimg)


def colorHist(path, windowNr):
    prevRoot = ""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                continue
            if not root == prevRoot:
                prevRoot = root
                print("We are in: "+ os.path.basename(os.path.normpath(root)))
                
            filePath = os.path.join(root,name)
            im = cv2.imread(filePath,1)
            HSV_im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            

            stride = int(128/windowNr)
            for i in range(windowNr):
                for j in range(windowNr):
                    window = im[i*stride:(i+1)*stride,j*stride:(j+1)*stride]
                    '''
                    ----Debug----
                    print(window[:,:,0].shape)
                    cv2.imshow("Window",window)
                    cv2.waitKey(0)
                    ----Debug----
                    '''
                    hHist = np.histogram(window[:,:,0])
                    sHist = np.histogram(window[:,:,1])
                    vHist = np.histogram(window[:,:,2])
                    hsvHist = np.concatenate((hHist,sHist,vHist))
                    print(hHist)
                    plt.hist(window[:,:,0].flatten(), bins='auto')
                    plt.title("Hue Histogram")
                    plt.show()
                    cv2.imshow("Window",window)
                    cv2.waitKey(0)
                    saveName = name.replace(".jpg","_"+str(windowNr)+"_"+str(i)+str(j)+".npy")
                    np.save(os.path.join(root,saveName),hsvHist)
                    

            
    print(os.path.basename(path))
    return 0
    
      
if __name__=="__main__":
    #inPath="C:\Users\Administrator\Downloads\Vispera-SKU101-2019\SKU_Recognition_Dataset\confectionery"
    resizeBatch(inPath)
