import utilities

if __name__=="__main__":
    confectionery = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\confectionery"
    icecream = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\icecream"
    laundry = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\laundry"
    soft1 = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\softdrinks-I"
    soft2 = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\softdrinks-II"

    utilities.resizeBatch(confectionery)
    print("20%")
    utilities.resizeBatch(icecream)
    print("40%")
    utilities.resizeBatch(laundry)
    print("60%")
    utilities.resizeBatch(soft1)
    print("80%")
    utilities.resizeBatch(soft2)
    print("Done!")

