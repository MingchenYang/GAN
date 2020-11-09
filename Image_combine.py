import os
import cv2
import numpy as np

Input_dir1 = 'S:/UCSD_ped2/Test256/Unet_Mosaic_test_diff/'
Input_dir2 = 'S:/UCSD_ped2/Test256/Unet_Reverse_test_diff/'
Output_dir = 'S:/UCSD_ped2/Test256/Unet_Mosaic_Reverse_com_test_diff/'

Input_name1 = os.listdir(Input_dir1)
Input_name2 = os.listdir(Input_dir2)

weights1 = 0.2
weights2 = 0.8


def read_and_load(path):
    img = cv2.imread(path, 0)
    img = img / 255.
    return img


def train():
    for i in range(len(Input_name1)):
        name1 = Input_name1[i]
        name2 = Input_name2[i]
        print(name1)

        Input_path1 = Input_dir1 + name1
        Input_path2 = Input_dir2 + name2

        image1 = read_and_load(Input_path1)
        image2 = read_and_load(Input_path2)

        image = weights1 * image1 + weights2 * image2

        cv2.imwrite(Output_dir + name1, image * 255.)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()