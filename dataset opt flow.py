import os
import cv2
import numpy as np

# input_dir = 'S:/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
#input_dir = 'S:/optical flow/MCL/Test/'
input_dir = 'S:/UCSD_ped2/Test/optical/'
#output_dir = 'S:/pix2pix optflow/Test/label/'
output_dir = 'S:/UCSD_ped2/Test/label/'

input_file = os.listdir(input_dir)
#input_file = input_file[2:]
#input_file.sort(key=lambda x: int(x[-3:]))

k = 0
for file_name in input_file:
    # Get the pictures name in files
    picture = os.listdir(input_dir + file_name + '/')
    num = 1
    # Change the path of training dataset
    for i in picture:
        path = input_dir + file_name + '/' + i
        print(path)
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        num1 = '%04d' % (k + 1)
        print(output_dir + str(num1) + '.jpg')
        cv2.imwrite(output_dir + str(num1) + '.jpg', x)
        k = k + 1
        num = num + 1
        if num == 200:
            break
