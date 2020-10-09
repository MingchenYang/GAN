import os
import cv2
import numpy as np

Input_dir = 'S:/pix2pix optflow/Test256/label/'
Output_dir = 'S:/pix2pix optflow/Test256 recover/label/'


def recover(img):
    max_val = np.max(img)
    img = (255. / max_val) * img
    return img


for file_name in os.listdir(Input_dir):
    file_path = Input_dir + file_name
    x = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    x_recover = recover(x)
    cv2.imwrite(Output_dir + file_name, x_recover)
    print(Output_dir + file_name)