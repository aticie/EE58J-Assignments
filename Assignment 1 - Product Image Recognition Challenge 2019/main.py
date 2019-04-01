import os
import sys
import argparse

import util

from tkinter import *
from tkinter import filedialog

cwd = os.getcwd()


def doEverything():
    global folder_path
    global sr_var
    global sc_var
    global shog_var
    global wsSetBox
    global binSetBox

    mainFolder = folder_path.get()
    sr = sr_var.get()
    sc = sc_var.get()
    shog = shog_var.get()
    ws = int(wsSetBox.get())
    bin_num = int(binSetBox.get())

    print(sr)
    print(sc)
    print(shog)
    try:
        print(ws)
    except:
        print("Please enter an integer value for window size!")
        return 0

    if not 128 % ws == 0:
        print("Sorry, 128 is not divisible by " + str(ws))
        print("Please select an appropriate window amount")
        return 0
    if ws <= 0:
        print("Window amount cannot be zero or negative.")
        print("Setting for 1 window per image.")
        ws = 1
    '''mainFolder = "C:\\EE58J - Data Mining for Visual Media\\Vispera-SKU101-2019\\SKU_Recognition_Dataset"

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
    confectionery = os.path.join(mainFolder, "confectionery")
    icecream = os.path.join(mainFolder, "icecream")
    laundry = os.path.join(mainFolder, "laundry")
    soft1 = os.path.join(mainFolder, "softdrinks-I")
    soft2 = os.path.join(mainFolder, "softdrinks-II")
    '''

    if sr:

        print("Resizing images skipped! (Assuming they are already resized)")

    else:

        print("Resizing images...")
        print("This usually takes 1 minute")
        util.resizeBatch(mainFolder)
        '''
        print("[0%]-----------")
        util.resizeBatch(confectionery)
        print("--[20%]--------")
        util.resizeBatch(icecream)
        print("----[40%]------")
        util.resizeBatch(laundry)
        print("------[60%]----")
        util.resizeBatch(soft1)
        print("--------[80%]--")
        util.resizeBatch(soft2)
        print("---------[100%]")
        print("Resizing Done!")
        '''
    if sc:
        print("Color Histogram part skipped!")

    else:
        '''
        print("Creating color histograms...")
        print("[0%]-----------")
        util.colorHist(confectionery, ws, bin_num)
        print("--[20%]--------")
        util.colorHist(icecream, ws, bin_num)
        print("----[40%]------")
        util.colorHist(laundry, ws, bin_num)
        print("------[60%]----")
        util.colorHist(soft1, ws, bin_num)
        print("--------[80%]--")
        util.colorHist(soft2, ws, bin_num)
        print("---------[100%]")
        print("Color histograms created!")
        '''
        print("Creating color histograms...")
        util.colorHist(mainFolder, ws, bin_num)

    if shog:
        print("HOG Histogram part skipped!")

    else:
        print("Creating Gradient Orientation histograms...")
        util.HOGHist(mainFolder, ws, bin_num)
        '''
        print("Creating Gradient Orientation histograms...")
        print("[0%]-----------")
        util.HOGHist(confectionery, ws, bin_num)
        print("--[20%]--------")
        util.HOGHist(icecream, ws, bin_num)
        print("----[40%]------")
        util.HOGHist(laundry, ws, bin_num)
        print("------[60%]----")
        util.HOGHist(soft1, ws, bin_num)
        print("--------[80%]--")
        util.HOGHist(soft2, ws, bin_num)
        print("---------[100%]")
        print("Gradient Orientation histograms created!")
        '''
    return 0


def classifier_run():
    global classifier_type
    global folder_path

    main_folder = folder_path.get()

    k = classifier_type.get()

    if k == 1:
        util.nnColor(main_folder)
    elif k == 2:
        util.nnOrient(main_folder)
    elif k == 3:
        util.nnCombine(main_folder)
    else:
        print("you shouldn't be able to do that...")
        return 0

    return 0


def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)


if __name__ == "__main__":
    mainFolder = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset"

    master = Tk()
    # ----- Image Description Extraction Buttons -----
    folder_path = StringVar()
    folder_path.set(mainFolder)
    lbl1 = Label(master, textvariable=folder_path)
    lbl1.grid(row=1, column=0)
    button2 = Button(text="Browse", command=browse_button)
    button2.grid(row=0, column=1)

    Label(master, text="Dataset root path").grid(row=0, sticky=W)

    sr_var = IntVar()
    Checkbutton(master, text="Skip Resize", variable=sr_var).grid(row=2, sticky=W)
    sc_var = IntVar()
    Checkbutton(master, text="Skip Color Hist.", variable=sc_var).grid(row=3, sticky=W)
    shog_var = IntVar()
    Checkbutton(master, text="Skip HOG", variable=shog_var).grid(row=4, sticky=W)
    Label(master, text="Window Size").grid(row=2, sticky=E)
    Label(master, text="Window per row/col").grid(row=3, column=1, sticky=E)
    Label(master, text="Ex. 1 for 1x1, 2 for 2x2").grid(row=4, column=1, sticky=E)
    wsSetBox = Entry(master)
    wsSetBox.insert(0, 1)
    wsSetBox.grid(row=2, column=1)
    binSetBox = Entry(master)
    binSetBox.insert(0, 10)
    binSetBox.grid(row=5, column=1)
    Label(master, text="Histogram Bins").grid(row=5, sticky=E)

    Button(master, text='Quit', command=master.quit).grid(row=5, column=0, sticky=W, pady=4)
    Button(master, text='Run Image Description Process', command=doEverything).grid(row=5, column=0, sticky=W, padx=45,
                                                                                    pady=4)

    # ----- Classification Buttons -----

    Label(master, text="----- Classification -----").grid(row=6, sticky=W)
    Button(master, text='Quit', command=master.quit).grid(row=8, column=0, sticky=W, pady=4)
    Button(master, text='Run Classifier', command=classifier_run).grid(row=8, column=0, sticky=W, padx=45, pady=4)
    classifier_type = IntVar()

    Radiobutton(master, text="NN - Color", variable=classifier_type, value=1).grid(row=7, sticky=W)
    Radiobutton(master, text="NN - Orient", variable=classifier_type, value=2).grid(row=7, sticky=W, padx=85)
    Radiobutton(master, text="NN - Combine", variable=classifier_type, value=3).grid(row=7, sticky=W, padx=175)

    mainloop()
