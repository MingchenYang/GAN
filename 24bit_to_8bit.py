import os
import cv2

Input_dir = 'S:/UCSD_ped2/Test256/training_Multi_frame_removal/'
Output_dir = 'S:/UCSD_ped2/Test256/training_Multi_frame_removal/'

Input_name = os.listdir(Input_dir)

for name in Input_name:
    image = cv2.imread(Input_dir + name, 0)
    cv2.imwrite(Output_dir + name, image)
    print(name)