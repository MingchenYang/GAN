import os
import cv2
import numpy as np

L_node = 256
W_node = 256

Input_dir = 'S:/UCSD_ped1/Test256/label/'
Output_dir = 'S:/UCSD_ped1/Test256/label0.75/'

for name in os.listdir(Input_dir):
    Input_path = Input_dir + name
    image = cv2.imread(Input_path, cv2.IMREAD_GRAYSCALE)
    image = image / 255.

    image1 = image[int(L_node * 0.25):L_node, :]
    cv2.imwrite(Output_dir + name, image1 * 255.)
