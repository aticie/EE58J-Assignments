import os
import sys
import argparse

import util

cwd = os.getcwd()

if __name__=="__main__":

    mainFolder = "C:\\EE58J - Data Mining for Visual Media\\Vispera-SKU101-2019\\SKU_Recognition_Dataset"

    parser = argparse.ArgumentParser(description='Main script for Assignment 1.')

    parser.add_argument('-sc', '--skipColor',
                        help="Skip Color Histogram (use if you have .npy files that contain color hist)",
                        action="store_true")

    parser.add_argument('Data', help="Full path to the dataset's root folder")
    
    parser.add_argument('-sr','--skipResize', help="Skip Resize (use if you have resized images)",
                        action="store_true")

    parser.add_argument('-shog', '--skipHog',
                        help="Skip HOG (Histogram of oriented gradients method)",
                        action="store_true")
    
    args = parser.parse_args()
    
    mainFolder = args.Data

    print(mainFolder)
    
    confectionery = os.path.join(mainFolder, "confectionery")
    icecream = os.path.join(mainFolder, "icecream")
    laundry = os.path.join(mainFolder, "laundry")
    soft1 = os.path.join(mainFolder, "softdrinks-I")
    soft2 = os.path.join(mainFolder, "softdrinks-II")

    if args.skipResize:

        print("Resizing images skipped! (Assuming they are already resized)")
        
    else:

        print("Resizing images...")
        print("This usually takes 1 minute")
        util.resizeBatch(confectionery)
        print("20%")
        util.resizeBatch(icecream)
        print("40%")
        util.resizeBatch(laundry)
        print("60%")
        util.resizeBatch(soft1)
        print("80%")
        util.resizeBatch(soft2)
        print("Done!")

    if args.skipColor:
        print("Color Histogram part skipped!")

    else:
        inputCorrect = False
        while not inputCorrect:
            ws = eval(input("Choose the window amount per image: "))
            if not 128%ws == 0:
                print("Sorry, 128 is not divisible by "+str(ws))
                print("Please select an appropriate window amount")
                continue
            if ws <= 0:
                print("Window amount cannot be zero or negative.")
                continue
            inputCorrect = True

        print("Creating color histograms...")
        print("[0%]-----------")
        util.colorHist(confectionery, ws)
        print("--[20%]--------")
        util.colorHist(icecream, ws)
        print("----[40%]------")
        util.colorHist(laundry, ws)
        print("------[60%]----")
        util.colorHist(soft1, ws)
        print("--------[80%]--")
        util.colorHist(soft2, ws)
        print("---------[100%]")
