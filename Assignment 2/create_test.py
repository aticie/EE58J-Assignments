import os
import math
import random

cwd = os.getcwd()

def create_test(path):
    save_path = os.path.join(path, "test")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dirlist = os.listdir(path)

    for folder in dirlist:
        # Path for 6973, 6574 class names
        classes = os.path.join(path, folder)

        file_list = os.listdir(classes)

        files = [name for name in file_list if name.endswith(".npy")]

        test_amount = math.floor((len(files)) * 0.2)

        print("Total files: " + str(len(files)))
        print("Test files: " + str(test_amount))

        chosen_samples = random.sample(files, k=test_amount)

        for item in chosen_samples:

            current_file_path = os.path.join(classes, item)

            save_path_class = os.path.join(save_path, folder)
            if not os.path.exists(save_path_class):
                os.mkdir(save_path_class)

            save_path_item = os.path.join(save_path_class, item)

            os.rename(current_file_path, save_path_item)


folder = os.path.join(cwd, "8x8")
create_test(folder)