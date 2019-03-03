import cv2
import os

def resizeBatch(inPath):
    for root, dirs, files in os.walk(inPath, topdown=False):
        for name in files:
            if not name.endswith(".jpg"):
                print(name+" - Doesn't end with .jpg")
                continue
            filePath = os.path.join(root,name)
            # Read image in RGB
            im = cv2.imread(filePath,1)
            # If image is already 128x128, do nothing
            if im.shape == (128,128,3):
                continue
            # Resize image to 128x128
            newimg = cv2.resize(im,(128,128))
            # Overwrite resized image
            cv2.imwrite(filePath,newimg)

def colorHist(image):
    
         
if __name__=="__main__":
    #inPath="C:\Users\Administrator\Downloads\Vispera-SKU101-2019\SKU_Recognition_Dataset\confectionery"
    resizeBatch(inPath)
