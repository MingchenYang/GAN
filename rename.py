import os
import cv2

Input_dir = 'S:/UCSD_ped2/Test256/training_Multi_frame_removal_unsorted/'
Output_dir = 'S:/UCSD_ped2/Test256/training_Multi_frame_removal/'

Input_name = os.listdir(Input_dir)

k = 1

for name in Input_name:
    num = '%04d' % k
    os.rename(Input_dir + name, Output_dir + str(num) + '.png')
    print(name)
    k = k + 1