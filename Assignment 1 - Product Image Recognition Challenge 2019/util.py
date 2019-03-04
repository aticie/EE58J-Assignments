import cv2
import os

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
            #print(root)
            filePath = os.path.join(root,name)
            # Read image in RGB
            im = cv2.imread(filePath,1)
            # If image is already 128x128, do nothing
            if im.shape == (128,128,3):
                #print(name+" is already resized!")
                continue
            # Resize image to 128x128
            # print("Resizing: "+name)
            # print("Dimensions: "+im.shape)
            newimg = cv2.resize(im,(128,128))
            # Overwrite resized image
            cv2.imwrite(filePath,newimg)

'''
#def colorHist(image, windowNr):
    if not 128%windowNr == 0:
        print("Sorry, 128 is not divisible by "+windowNr")
        print("Please select an appropriate Window amount")
        return 0
    
'''      
if __name__=="__main__":
    #inPath="C:\Users\Administrator\Downloads\Vispera-SKU101-2019\SKU_Recognition_Dataset\confectionery"
    resizeBatch(inPath)
