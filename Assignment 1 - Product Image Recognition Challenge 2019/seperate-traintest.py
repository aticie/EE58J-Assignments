import os
import random
import math
#random.sample(range(100), 10)
def create_test(path):

    save_path = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset\\test"

    main_folder = os.path.basename(os.path.normpath(path))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Path for confectionery, softdrinks .. etc
    dirlist = os.listdir(path)

    for folder in dirlist:

        # Path for 6973, 6574 class names
        classes = os.path.join(path, folder)

        if not os.path.isdir(classes):
            continue

        file_list = os.listdir(classes)

        files = [name for name in file_list if name.endswith(".jpg")]

        test_amount = math.floor((len(files))*0.2)

        chosen_samples = random.sample(files, k=test_amount)

        for item in chosen_samples:

            color = item.replace(".jpg", "_color.npy")
            orient = item.replace(".jpg", "_orient.npy")

            current_file_path = os.path.join(classes, item)
            current_color_path = os.path.join(classes, color)
            current_orient_path = os.path.join(classes, orient)


            save_path_folder = os.path.join(save_path, main_folder)
            if not os.path.exists(save_path_folder):
                os.mkdir(save_path_folder)
            save_path_class = os.path.join(save_path_folder, folder)
            if not os.path.exists(save_path_class):
                os.mkdir(save_path_class)

            save_path_item = os.path.join(save_path_class, item)
            save_path_item_color = os.path.join(save_path_class, color)
            save_path_item_orient = os.path.join(save_path_class, orient)


            os.rename(current_file_path, save_path_item)
            os.rename(current_color_path, save_path_item_color)
            os.rename(current_orient_path, save_path_item_orient)



    return 0


path = "C:\\Users\\Administrator\\Downloads\\Vispera-SKU101-2019\\SKU_Recognition_Dataset"

confectionery = os.path.join(path, "confectionery")
icecream = os.path.join(path, "icecream")
laundry = os.path.join(path, "laundry")
soft1 = os.path.join(path, "softdrinks-I")
soft2 = os.path.join(path, "softdrinks-II")

create_test(confectionery)
create_test(icecream)
create_test(laundry)
create_test(soft1)
create_test(soft2)

dirlist = os.listdir(path)
