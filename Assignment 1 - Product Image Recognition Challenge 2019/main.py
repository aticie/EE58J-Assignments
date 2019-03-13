import os
import sys
import argparse

import util

from tkinter import *
from tkinter import filedialog


cwd = os.getcwd()

def doEverything(data,sr,sc,shog,ws):
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
    
    mainFolder = args.Data'''

    mainFolder = data

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

    if args.skipHog:
        print("HOG Histogram part skipped!")
        
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
        util.HOGHist(confectionery,ws)

def show_entry_fields():
   print("Path: %s\n" % e1.get())

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

def var_states():
   print("Skip Resize: %d,\nSkip Color Hist.: %d\nSkip HOG: %d\n" % (var1.get(), var2.get(),var3.get()))
    
if __name__=="__main__":

    master = Tk()

    folder_path = StringVar()
    lbl1 = Label(master,textvariable=folder_path)
    lbl1.grid(row=0, column=1)
    button2 = Button(text="Browse", command=browse_button)
    button2.grid(row=0, column=3)

    Label(master, text="Dataset root path").grid(row=0)

    T = Text(master,height=1,width=50)
    T.grid(row=1)
    T.insert(END, cwd)

    var1 = IntVar()
    Checkbutton(master, text="Skip Resize", variable=var1).grid(row=2, sticky=W)
    var2 = IntVar()
    Checkbutton(master, text="Skip Color Hist.", variable=var2).grid(row=3, sticky=W)
    var3 = IntVar()
    Checkbutton(master, text="Skip HOG", variable=var3).grid(row=4, sticky=W)
    Button(master, text='Quit', command=master.quit).grid(row=5,column=0, sticky=W, pady=4)
    Button(master, text='Show', command=var_states).grid(row=5, column=0, sticky=W, padx=45, pady=4)

    mainloop()


    
    
