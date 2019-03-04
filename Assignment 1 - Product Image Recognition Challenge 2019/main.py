import os
import util

if __name__=="__main__":

    mainFolder = "C:\\EE58J - Data Mining for Visual Media\\Vispera-SKU101-2019\\SKU_Recognition_Dataset"

    confectionery = os.path.join(mainFolder,"confectionery")
    icecream = os.path.join(mainFolder,"icecream")
    laundry = os.path.join(mainFolder,"laundry")
    soft1 = os.path.join(mainFolder,"softdrinks-I")
    soft2 = os.path.join(mainFolder,"softdrinks-II")

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
