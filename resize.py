import os
import cv2

Input_dir = 'S:/UCSD_ped2/Test/training_full_removal/'
Output_dir = 'S:/UCSD_ped2/Test256/training_full_removal/'

for file_name in os.listdir(Input_dir):
    file_path = Input_dir + file_name
    x = cv2.imread(file_path, 0)
    x_resize = cv2.resize(x, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(Output_dir + file_name, x_resize)
    print(Output_dir + file_name)