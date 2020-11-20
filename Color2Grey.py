# Colorful 3-channel images to greyscale images
import os
import cv2

Input_dir = 'S:/CUHK/Train/training_color/'
Output_dir= 'S:/CUHK/Train/training/'

Input_name = os.listdir(Input_dir)

for name in Input_name:
    image = cv2.imread(Input_dir + name, 0)
    print(name)
    cv2.imwrite(Output_dir + name, image)